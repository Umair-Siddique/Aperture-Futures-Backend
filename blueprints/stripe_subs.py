from flask import Blueprint, request, jsonify, current_app
import stripe
from functools import wraps
from blueprints.auth import token_required
import traceback

subscription_bp = Blueprint('subscription', __name__)

@subscription_bp.route('/plans', methods=['GET'])
def get_subscription_plans():
    """
    Get available subscription plans with pricing information
    Public endpoint - no authentication required
    """
    try:
        # Get price ID from config
        price_id = current_app.config.get('STRIPE_PRICE_ID')
        
        if not price_id:
            from config import Config
            price_id = Config.STRIPE_PRICE_ID
        
        if not price_id:
            return jsonify({'error': 'Stripe price ID not configured'}), 500
        
        # Retrieve price details from Stripe
        price = stripe.Price.retrieve(price_id, expand=['product'])
        
        plan = {
            'id': price.id,
            'currency': price.currency,
            'unit_amount': price.unit_amount,  # Amount in cents
            'interval': price.recurring.interval if price.recurring else None,
            'interval_count': price.recurring.interval_count if price.recurring else None,
            'product': {
                'id': price.product.id,
                'name': price.product.name,
                'description': price.product.description,
            }
        }
        
        return jsonify({'plan': plan}), 200
        
    except stripe.error.StripeError as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@subscription_bp.route('/create-checkout-session', methods=['POST'])
@token_required
def create_checkout_session(user):
    """
    Create a Stripe Checkout Session for subscription
    User must be authenticated
    """
    try:
        # Try to get from app config first
        price_id = current_app.config.get('STRIPE_PRICE_ID')
        
        # Fallback: try to get directly from Config class
        if not price_id:
            from config import Config
            price_id = Config.STRIPE_PRICE_ID
        
        if not price_id:
            return jsonify({
                'error': 'Stripe price ID not configured',
                'message': 'STRIPE_PRICE_ID environment variable is not set. Please set it in your environment variables.',
                'hint': 'Add STRIPE_PRICE_ID=price_YOUR_PRICE_ID to your environment variables (e.g., in Render dashboard or .env file)'
            }), 500
        
        # Try to get from app config first
        frontend_url = current_app.config.get('FRONTEND_URL')
        
        # Fallback: try to get directly from Config class
        if not frontend_url:
            from config import Config
            frontend_url = Config.FRONTEND_URL
        
        # Final fallback to default
        if not frontend_url:
            frontend_url = 'http://localhost:5173'
        
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
        
    except stripe.error.InvalidRequestError as e:
        # Check if it's a price ID error
        error_msg = str(e)
        if 'No such price' in error_msg:
            traceback.print_exc()
            return jsonify({
                'error': 'Invalid Stripe price ID',
                'message': 'The price ID does not exist in your current Stripe environment. Make sure you are using a test price ID when using test/sandbox keys, or a production price ID when using production keys.',
                'hint': 'Check your STRIPE_PRICE_ID environment variable and ensure it matches your Stripe environment (test vs production)',
                'original_error': error_msg
            }), 400
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@subscription_bp.route('/verify-checkout-session', methods=['GET'])
@token_required
def verify_checkout_session(user):
    """
    Verify a checkout session after payment
    Used on the success page to confirm payment and get subscription details
    """
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id parameter is required'}), 400
        
        # Retrieve the session from Stripe
        session = stripe.checkout.Session.retrieve(
            session_id,
            expand=['subscription', 'customer']
        )
        
        # Verify the session belongs to the authenticated user
        session_user_id = session.metadata.get('user_id')
        if session_user_id != user.id:
            return jsonify({'error': 'Unauthorized access to this session'}), 403
        
        # Get subscription details if available
        subscription_info = None
        subscription_id = None
        subscription_status = None
        if session.subscription:
            # Handle both expanded subscription object and subscription ID string
            if isinstance(session.subscription, str):
                # If it's just an ID, retrieve the full subscription
                subscription = stripe.Subscription.retrieve(session.subscription)
            else:
                # If it's already expanded, use it directly
                subscription = session.subscription
            
            # Safely get subscription fields (some may not exist immediately after checkout)
            subscription_id = subscription.id
            subscription_status = getattr(subscription, 'status', None)
            subscription_info = {
                'id': subscription.id,
                'status': subscription_status,
                'current_period_start': getattr(subscription, 'current_period_start', None),
                'current_period_end': getattr(subscription, 'current_period_end', None),
                'cancel_at_period_end': getattr(subscription, 'cancel_at_period_end', False),
            }

        # Persist subscription status so /subscription-status reflects immediately
        if session.payment_status == 'paid' and session.customer:
            try:
                current_app.supabase_admin.table('profiles').upsert({
                    'id': user.id,
                    'stripe_customer_id': session.customer,
                    'subscription_id': subscription_id,
                    'subscription_status': subscription_status or 'active',
                    'role': 'user'
                }).execute()
            except Exception as e:
                print(f"Error updating profile after checkout verification for {user.id}: {e}")
        
        return jsonify({
            'success': True,
            'session_id': session.id,
            'payment_status': session.payment_status,
            'customer_email': session.customer_email,
            'subscription': subscription_info
        }), 200
        
    except stripe.error.StripeError as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
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
        profile_response = None
        try:
            profile_response = current_app.supabase_admin.table('profiles').select('stripe_customer_id, subscription_status, subscription_id').eq('id', user.id).execute()
        except Exception as e:
            # Profile doesn't exist - return default status
            print(f"Profile not found for user {user.id}: {e}")
            return jsonify({
                'has_subscription': False,
                'status': 'none'
            }), 200
        
        if not profile_response or not profile_response.data:
            return jsonify({
                'has_subscription': False,
                'status': 'none'
            }), 200
        # Supabase returns a list for non-single queries
        profile = profile_response.data[0] if isinstance(profile_response.data, list) else profile_response.data
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
                    'status': getattr(subscription, 'status', None),
                    'current_period_end': getattr(subscription, 'current_period_end', None),
                    'cancel_at_period_end': getattr(subscription, 'cancel_at_period_end', False),
                    'current_period_start': getattr(subscription, 'current_period_start', None),
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
                # Upsert user profile with subscription info (creates if doesn't exist, updates if exists)
                try:
                    current_app.supabase_admin.table('profiles').upsert({
                        'id': user_id,
                        'stripe_customer_id': customer_id,
                        'subscription_id': subscription_id,
                        'subscription_status': 'active',
                        'role': 'user'  # Default role if creating new profile
                    }).execute()
                    print(f"✓ Subscription activated for user {user_id}")
                except Exception as e:
                    print(f"Error updating profile for user {user_id}: {e}")
                    # Try update as fallback
                    try:
                        current_app.supabase_admin.table('profiles').update({
                            'stripe_customer_id': customer_id,
                            'subscription_id': subscription_id,
                            'subscription_status': 'active'
                        }).eq('id', user_id).execute()
                        print(f"✓ Subscription activated for user {user_id} (via update)")
                    except Exception as update_error:
                        print(f"Failed to update profile: {update_error}")
        
        elif event_type == 'customer.subscription.updated':
            # Subscription status changed
            subscription = data
            customer_id = subscription.get('customer')
            subscription_status = subscription.get('status')
            subscription_id = subscription.get('id')
            
            # Find user by customer_id
            profile_response = current_app.supabase_admin.table('profiles').select('id').eq('stripe_customer_id', customer_id).single().execute()
            
            if profile_response.data:
                user_id = profile_response.data['id']
                current_app.supabase_admin.table('profiles').update({
                    'subscription_status': subscription_status,
                    'subscription_id': subscription_id
                }).eq('id', user_id).execute()
                print(f"✓ Subscription updated for user {user_id}: {subscription_status}")
        
        elif event_type == 'customer.subscription.deleted':
            # Subscription cancelled
            subscription = data
            customer_id = subscription.get('customer')
            
            profile_response = current_app.supabase_admin.table('profiles').select('id').eq('stripe_customer_id', customer_id).single().execute()
            
            if profile_response.data:
                user_id = profile_response.data['id']
                current_app.supabase_admin.table('profiles').update({
                    'subscription_status': 'cancelled',
                    'subscription_id': None
                }).eq('id', user_id).execute()
                print(f"✓ Subscription cancelled for user {user_id}")
        
        elif event_type == 'invoice.payment_failed':
            # Payment failed
            invoice = data
            customer_id = invoice.get('customer')
            
            profile_response = current_app.supabase_admin.table('profiles').select('id').eq('stripe_customer_id', customer_id).single().execute()
            
            if profile_response.data:
                user_id = profile_response.data['id']
                current_app.supabase_admin.table('profiles').update({
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
        profile_response = current_app.supabase_admin.table('profiles').select('subscription_id').eq('id', user.id).single().execute()
        
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
        profile_response = current_app.supabase_admin.table('profiles').select('subscription_id').eq('id', user.id).single().execute()
        
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

@subscription_bp.route('/invoices', methods=['GET'])
@token_required
def get_invoices(user):
    """
    Get invoice history for the authenticated user
    """
    try:
        # Get customer ID from profile
        profile_response = current_app.supabase_admin.table('profiles').select('stripe_customer_id').eq('id', user.id).single().execute()
        
        if not profile_response.data or not profile_response.data.get('stripe_customer_id'):
            return jsonify({'invoices': []}), 200
        
        customer_id = profile_response.data['stripe_customer_id']
        
        # Get invoices from Stripe
        invoices = stripe.Invoice.list(customer=customer_id, limit=10)
        
        invoice_list = []
        for invoice in invoices.data:
            invoice_list.append({
                'id': invoice.id,
                'amount_due': invoice.amount_due,
                'amount_paid': invoice.amount_paid,
                'currency': invoice.currency,
                'status': invoice.status,
                'created': invoice.created,
                'period_start': invoice.period_start,
                'period_end': invoice.period_end,
                'invoice_pdf': invoice.invoice_pdf,
                'hosted_invoice_url': invoice.hosted_invoice_url,
            })
        
        return jsonify({'invoices': invoice_list}), 200
        
    except stripe.error.StripeError as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
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
        profile_response = current_app.supabase_admin.table('profiles').select('stripe_customer_id').eq('id', user.id).single().execute()
        
        if not profile_response.data or not profile_response.data.get('stripe_customer_id'):
            return jsonify({'error': 'No Stripe customer found'}), 400
        
        customer_id = profile_response.data['stripe_customer_id']
        
        # Try to get from app config first
        frontend_url = current_app.config.get('FRONTEND_URL')
        
        # Fallback: try to get directly from Config class
        if not frontend_url:
            from config import Config
            frontend_url = Config.FRONTEND_URL
        
        # Final fallback to default
        if not frontend_url:
            frontend_url = 'http://localhost:5173'
        
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

