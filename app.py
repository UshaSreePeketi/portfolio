import streamlit as st
import requests

st.title("Best Product Type to sell")
st.write("By Usha Sree Peketi")
st.write("Enter the details to check which product to sell!")

# Input Fields for the API
area_code = st.number_input("Area Code", min_value=200, max_value=999, step=1)

# Providing the dropdown options for categorical columns
state = st.selectbox("State", ['Connecticut', 'Washington', 'California', 'Texas', 'New York', 'Ohio',
                               'Illinois', 'Louisiana', 'Florida', 'Wisconsin', 'Colorado', 'Missouri', 
                               'Iowa', 'Massachusetts', 'Oklahoma', 'Utah', 'Oregon', 'New Mexico', 
                               'New Hampshire', 'Nevada'])

market = st.selectbox("Market", ['East', 'West', 'South', 'Central'])

market_size = st.selectbox("Market Size", ['Small Market', 'Major Market'])

product = st.selectbox("Product", ['Columbian', 'Green Tea', 'Caffe Mocha', 'Decaf Espresso', 'Lemon', 'Mint',
                                  'Darjeeling', 'Decaf Irish Cream', 'Chamomile', 'Earl Grey', 'Caffe Latte',
                                  'Amaretto', 'Regular Espresso'])

# Input Fields for numerical data
sales = st.number_input("Sales", min_value=0.0, step=0.01)
profit = st.number_input("Profit", min_value=0.0, step=0.01)
cogs = st.number_input("COGS", min_value=0.0, step=0.01)
marketing = st.number_input("Marketing", min_value=0.0, step=0.01)
total_expenses = st.number_input("Total Expenses", min_value=0.0, step=0.01)
margin = st.number_input("Margin", min_value=0.0, step=0.01)

# Button to Trigger Prediction
if st.button("Classify"):
    # Validate input to ensure all fields are filled
    if state and market and market_size and product and area_code and sales is not None and profit is not None and cogs is not None and marketing is not None and total_expenses is not None and margin is not None:
        try:
            api_url = "http://18.217.75.50:8000/predict/"
            
            # Construct the payload
            payload = {
                "data": {
                    "Area_Code": area_code,
                    "State": state,
                    "Market": market,
                    "Market_Size": market_size,
                    "Sales": sales,
                    "Profit": profit,
                    "COGS": cogs,
                    "Product": product,
                    "Marketing": marketing,
                    "Total_Expenses": total_expenses,
                    "Margin": margin
                }
            }
            
            # Make POST Request
            response = requests.post(api_url, json=payload)
            
            # Parse Response
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unable to get prediction from the API.')}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please fill out all required fields.")
else:
    st.info("Fill in the fields above and click 'Classify' to get a prediction.")

# Footer
st.write("Powered by Streamlit & FastAPI")
