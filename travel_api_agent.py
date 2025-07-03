import requests
import json

def get_travel_recommendations(prompt, customer_id=None):
    """Get complete travel data with both destination and itinerary"""
    # Stage 1: Get destination recommendations
    destination_data = get_destination_recommendations(prompt)
    
    # Stage 2: Get itinerary if destination exists and customer_id provided
    itinerary_data = None
    if destination_data and customer_id and destination_data.get('destination_id'):
        print(f"🗓️ Creating itinerary for destination {destination_data['destination_id']} and customer {customer_id}")
        itinerary_data = create_travel_itinerary(destination_data['destination_id'], customer_id)
    
    return {
        "destination": destination_data,
        "itinerary": itinerary_data
    }

def get_destination_recommendations(prompt):
    """Get destination recommendations from the travel API"""
    url = "https://travel-assistant-api.testmaya.com/recommend"
    
    # Set up headers
    headers = {
        'accept': 'application/json'
    }
    
    # Set up query parameters
    params = {
        'prompt': prompt
    }
    
    try:
        # Make the GET request
        response = requests.get(url, headers=headers, params=params)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Get the raw JSON response
        raw_data = response.json()
        
        # Parse and format the response into properly nested data
        formatted_data = format_travel_response(raw_data)
        
        return formatted_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error making destination request: {e}")
        return None

def create_travel_itinerary(destination_id, customer_id):
    """Create travel itinerary for a specific destination and customer"""
    url = f"https://travel-assistant-api.testmaya.com/itinerary"
    
    # Set up headers
    headers = {
        'accept': 'application/json'
    }
    
    # Set up query parameters
    params = {
        'destination_id': destination_id,
        'user_id': customer_id,
        'mode': 'create',
        'list_type': 'transaction'
    }
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, params=params, data='')
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Get the JSON response
        itinerary_data = response.json()
        
        print(f"✅ Itinerary created successfully for destination {destination_id}")
        return itinerary_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating itinerary: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing itinerary response: {e}")
        return None

def format_travel_response(raw_data):
    """Parse nested JSON strings into proper nested objects"""
    if not raw_data:
        return None
    
    try:
        # Parse temperature_details if it exists and is a string
        if 'temperature_details' in raw_data and isinstance(raw_data['temperature_details'], str):
            try:
                raw_data['temperature_details'] = json.loads(raw_data['temperature_details'])
            except json.JSONDecodeError:
                pass  # Keep original if parsing fails
        
        # Parse price_point if it exists and is a string
        if 'price_point' in raw_data and isinstance(raw_data['price_point'], str):
            try:
                raw_data['price_point'] = json.loads(raw_data['price_point'])
            except json.JSONDecodeError:
                pass  # Keep original if parsing fails
        
        # Parse destination_tags if it exists and is a string
        if 'destination_tags' in raw_data and isinstance(raw_data['destination_tags'], str):
            try:
                raw_data['destination_tags'] = json.loads(raw_data['destination_tags'])
            except json.JSONDecodeError:
                pass  # Keep original if parsing fails
        
        return raw_data
    
    except Exception as e:
        print(f"Error formatting travel response: {e}")
        return raw_data  # Return original data if formatting fails

if __name__ == "__main__":
    # Example usage
    prompt = "suggest me some beach destination for a 3 day vacation"
    customer_id = "58"
    
    print("🧪 Testing Complete Travel Data Retrieval")
    print("="*50)
    
    recommendations = get_travel_recommendations(prompt, customer_id)
    
    if recommendations:
        print("🏖️ Travel Recommendations:")
        print(f"📍 Destination: {recommendations.get('destination', {}).get('city', 'N/A')}")
        print(f"🗓️ Itinerary Available: {bool(recommendations.get('itinerary'))}")
        
        print(f"\n📋 Full Response:")
        print(json.dumps(recommendations, indent=2, ensure_ascii=False))
