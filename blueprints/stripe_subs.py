from flask import Blueprint, request, jsonify, current_app
import stripe
from functools import wraps
from blueprints.auth import token_required
import traceback

subscription_bp = Blueprint('subscription', __name__)

@subscription_bp.route('/create-checkout-session', methods=['POST'])
@token_required
def create_checkout_session(user):
    """
    Create a Stripe Checkout Session for subscription
    User must be authenticated
    """
    try:
        price_id = current_app.config.get('STRIPE_PRICE_ID')
        if not price_id:
            return jsonify({'error': 'Stripe price ID not configured'}), 500
        
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:5173')
        
        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            customer_email=user.email,
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f'{frontend_url}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{frontend_url}/subscription/cancel',
            metadata={
                'user_id': user.id,
            },
            subscription_data={
                'metadata': {
                    'user_id': user.id,
                }
            }
        )
        
        return jsonify({
            'checkout_url': checkout_session.url,
            'session_id': checkout_session.id
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@subscription_bp.route('/subscription-status', methods=['GET'])
@token_required
def get_subscription_status(user):
    """
    Get current subscription status for authenticated user
    """
    try:
        # Get customer ID from Supabase profiles table
        profile_response = current_app.supabase.table('profiles').select('stripe_customer_id, subscription_status, subscription_id').eq('id', user.id).single().execute()
        
        if not profile_response.data:
            return jsonify({
                'has_subscription': False,
                'status': 'none'
            }), 200
        
        profile = profile_response.data
        stripe_customer_id = profile.get('stripe_customer_id')
        subscription_status = profile.get('subscription_status', 'none')
        subscription_id = profile.get('subscription_id')
        
        if not stripe_customer_id:
            return jsonify({
                'has_subscription': False,
                'status': 'none'
            }), 200
        
        # Get subscription details from Stripe
        subscription_info = None
        if subscription_id:
            try:
                subscription = stripe.Subscription.retrieve(subscription_id)
                subscription_info = {
                    'status': subscription.status,
                    'current_period_end': subscription.current_period_end,
                    'cancel_at_period_end': subscription.cancel_at_period_end,
                    'current_period_start': subscription.current_period_start,
                }
            except stripe.error.StripeError as e:
                print(f"Error retrieving subscription from Stripe: {e}")
                pass
        
        return jsonify({
            'has_subscription': subscription_status in ['active', 'trialing'],
            'status': subscription_status,
            'subscription_info': subscription_info
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@subscription_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """
    Handle Stripe webhook events
    This endpoint should be publicly accessible (no auth required)
    """
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = current_app.config.get('STRIPE_WEBHOOK_SECRET')
    
    if not webhook_secret:
        return jsonify({'error': 'Webhook secret not configured'}), 500
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError as e:
        print(f"Invalid payload: {e}")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        print(f"Invalid signature: {e}")
        return jsonify({'error': 'Invalid signature'}), 400
    
    # Handle the event
    event_type = event['type']
    data = event['data']['object']
    
    try:
        if event_type == 'checkout.session.completed':
            # Payment successful, subscription created
            session = data
            user_id = session.get('metadata', {}).get('user_id')
            customer_id = session.get('customer')
            subscription_id = session.get('subscription')
            
            if user_id and customer_id:
                # Update user profile with subscription info
                current_app.supabase.table('profiles').update({
                    'stripe_customer_id': customer_id,
                    'subscription_id': subscription_id,
                    'subscription_status': 'active'
                }).eq('id', user_id).execute()
                print(f"✓ Subscription activated for user {user_id}")
        
        elif event_type == 'customer.subscription.updated':
            # Subscription status changed
            subscription = data
            customer_id = subscription.get('customer')
            subscription_status = subscription.get('status')
            subscription_id = subscription.get('id')
            
            # Find user by customer_id
            profile_response = current_app.supabase.table('profiles').select('id').eq('stripe_customer_id', customer_id).single().execute()
            
            if profile_response.data:
                user_id = profile_response.data['id']
                current_app.supabase.table('profiles').update({
                    'subscription_status': subscription_status,
                    'subscription_id': subscription_id
                }).eq('id', user_id).execute()
                print(f"✓ Subscription updated for user {user_id}: {subscription_status}")
        
        elif event_type == 'customer.subscription.deleted':
            # Subscription cancelled
            subscription = data
            customer_id = subscription.get('customer')
            
            profile_response = current_app.supabase.table('profiles').select('id').eq('stripe_customer_id', customer_id).single().execute()
            
            if profile_response.data:
                user_id = profile_response.data['id']
                current_app.supabase.table('profiles').update({
                    'subscription_status': 'cancelled',
                    'subscription_id': None
                }).eq('id', user_id).execute()
                print(f"✓ Subscription cancelled for user {user_id}")
        
        elif event_type == 'invoice.payment_failed':
            # Payment failed
            invoice = data
            customer_id = invoice.get('customer')
            
            profile_response = current_app.supabase.table('profiles').select('id').eq('stripe_customer_id', customer_id).single().execute()
            
            if profile_response.data:
                user_id = profile_response.data['id']
                current_app.supabase.table('profiles').update({
                    'subscription_status': 'past_due'
                }).eq('id', user_id).execute()
                print(f"✓ Payment failed for user {user_id}")
        
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@subscription_bp.route('/cancel-subscription', methods=['POST'])
@token_required
def cancel_subscription(user):
    """
    Cancel user's subscription at period end
    """
    try:
        # Get subscription ID from profile
        profile_response = current_app.supabase.table('profiles').select('subscription_id').eq('id', user.id).single().execute()
        
        if not profile_response.data or not profile_response.data.get('subscription_id'):
            return jsonify({'error': 'No active subscription found'}), 400
        
        subscription_id = profile_response.data['subscription_id']
        
        # Cancel subscription at period end
        subscription = stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=True
        )
        
        return jsonify({
            'message': 'Subscription will be cancelled at period end',
            'cancel_at_period_end': subscription.cancel_at_period_end
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@subscription_bp.route('/reactivate-subscription', methods=['POST'])
@token_required
def reactivate_subscription(user):
    """
    Reactivate a cancelled subscription
    """
    try:
        profile_response = current_app.supabase.table('profiles').select('subscription_id').eq('id', user.id).single().execute()
        
        if not profile_response.data or not profile_response.data.get('subscription_id'):
            return jsonify({'error': 'No subscription found'}), 400
        
        subscription_id = profile_response.data['subscription_id']
        
        # Reactivate subscription
        subscription = stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=False
        )
        
        return jsonify({
            'message': 'Subscription reactivated',
            'cancel_at_period_end': subscription.cancel_at_period_end
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@subscription_bp.route('/customer-portal', methods=['POST'])
@token_required
def create_customer_portal_session(user):
    """
    Create a Stripe Customer Portal session for managing subscription
    """
    try:
        # Get customer ID from profile
        profile_response = current_app.supabase.table('profiles').select('stripe_customer_id').eq('id', user.id).single().execute()
        
        if not profile_response.data or not profile_response.data.get('stripe_customer_id'):
            return jsonify({'error': 'No Stripe customer found'}), 400
        
        customer_id = profile_response.data['stripe_customer_id']
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:5173')
        
        # Create portal session
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f'{frontend_url}/subscription',
        )
        
        return jsonify({
            'url': portal_session.url
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

