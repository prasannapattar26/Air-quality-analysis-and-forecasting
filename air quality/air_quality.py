import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import calendar
from twilio.rest import Client

# Load the dataset
df = pd.read_csv('Air_data.csv')

# Clean and preprocess the data
df.dropna(inplace=True)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.day_name()
df['hour'] = df['date'].dt.hour

# Function to send an alert message to the city corporation number
def send_alert(city, pollution_level):
    # Twilio credentials (replace with your own)
    account_sid = 'ACdb9071da5b51d812e9e7cf586454f820'
    auth_token = '69c2864baee8d81fcfb1997bd3d8bd9d'
    client = Client(account_sid, auth_token)

    message = f"Alert! The predicted average pollution level in {city} for next year is {pollution_level:.2f}, exceeding safe limits."

    try:
        client.messages.create(
            body=message,
            from_='+16314898583',
            to='+919380663870'
        )
        print(f"Alert sent to {city} corporation.")
    except Exception as e:
        print(f"Failed to send alert to {city} corporation: {e}")

# Function to plot trends for selected cities with month names and AM/PM format for hours
def plot_city_trends(df, cities):
    sns.set(style="whitegrid")
    
    for city in cities:
        city_df = df[df['city'] == city]
        
        # Calculate trends for pollution level and each gas
        yearly_trends = city_df.groupby('year')[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()
        monthly_trends = city_df.groupby('month')[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()
        monthly_trends.index = monthly_trends.index.map(lambda x: calendar.month_name[x])
        weekly_trends = city_df.groupby('day_of_week')[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()
        
        # Convert hour to AM/PM format
        def convert_to_ampm(hour):
            if hour == 0:
                return '12 AM'
            elif 1 <= hour <= 11:
                return f'{hour} AM'
            elif hour == 12:
                return '12 PM'
            else:
                return f'{hour - 12} PM'
        
        hourly_trends = city_df.groupby(city_df['hour'].apply(convert_to_ampm))[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()
        
        regional_trends = city_df.groupby('region')[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()
        
        # Plotting
        plt.figure(figsize=(16, 10))
        
        # Plot yearly trends
        plt.subplot(231)  # 2 rows, 3 columns, subplot 1
        sns.lineplot(x=yearly_trends.index, y=yearly_trends['pollution_level'], marker='o', label='Overall')
        for gas in ['CO', 'SO2', 'NOx', 'O3']:
            sns.lineplot(x=yearly_trends.index, y=yearly_trends[gas], marker='o', label=gas)
        plt.title(f'Yearly Air Pollution Trends - {city}')
        plt.xlabel('Year')
        plt.ylabel('Average Pollution Level')
        plt.legend()
        
        # Plot monthly trends
        plt.subplot(232)  # subplot 2
        sns.lineplot(x=monthly_trends.index, y=monthly_trends['pollution_level'], marker='o', label='Overall')
        for gas in ['CO', 'SO2', 'NOx', 'O3']:
            sns.lineplot(x=monthly_trends.index, y=monthly_trends[gas], marker='o', label=gas)
        plt.title(f'Monthly Air Pollution Trends - {city}')
        plt.xlabel('Month')
        plt.ylabel('Average Pollution Level')
        plt.legend()
        
        # Plot weekly trends
        plt.subplot(234)  # subplot 3
        weekly_trends_melted = weekly_trends.reset_index().melt(id_vars='day_of_week', value_vars=['pollution_level', 'CO', 'SO2', 'NOx', 'O3'])
        sns.barplot(x='day_of_week', y='value', hue='variable', data=weekly_trends_melted)
        plt.title(f'Weekly Air Pollution Trends - {city}')
        plt.xlabel('Day of the Week')
        plt.ylabel('Average Pollution Level')
        plt.legend(title='Gas')
        
        # Plot hourly trends
        plt.subplot(233)  # subplot 4
        sns.lineplot(x=hourly_trends.index, y=hourly_trends['pollution_level'], marker='o', label='Overall')
        for gas in ['CO', 'SO2', 'NOx', 'O3']:
            sns.lineplot(x=hourly_trends.index, y=hourly_trends[gas], marker='o', label=gas)
        plt.title(f'Hourly Air Pollution Trends - {city}')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Average Pollution Level')
        plt.legend()
        
        # Plot regional trends
        plt.subplot(235)  # subplot 5
        regional_trends_melted = regional_trends.reset_index().melt(id_vars='region', value_vars=['pollution_level', 'CO', 'SO2', 'NOx', 'O3'])
        sns.barplot(x='region', y='value', hue='variable', data=regional_trends_melted)
        plt.title(f'Regional Air Pollution Levels - {city}')
        plt.xlabel('Region')
        plt.ylabel('Average Pollution Level')
        plt.xticks(rotation=45)
        plt.legend(title='Gas')
        
        plt.tight_layout()
        plt.show()

# Function to predict future pollution levels for each city using polynomial regression
def predict_future_pollution(df, cities):
    predictions = {}
    
    for city in cities:
        city_df = df[df['city'] == city]
        
        # Prepare data for prediction for pollution level and each gas
        X = city_df[['month']].values
        y = city_df['pollution_level'].values
        
        # Polynomial features for pollution level
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Train a polynomial regression model for pollution level
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predict next month's pollution levels for pollution level
        next_months = (city_df['month'].unique() % 12) + 1
        next_month_predictions = model.predict(poly.transform(next_months.reshape(-1, 1)))
        
        # Ensure predictions are non-negative and not zero
        next_month_predictions = np.maximum(next_month_predictions, 0.1)  # Adjust as needed
        
        # Predict next year's pollution level for pollution level
        avg_next_year_prediction = max(model.predict(poly.transform([[city_df['month'].max() + 12]]))[0], 0.1)  # Adjust as needed
        
        # Store predictions for pollution level
        if city not in predictions:
            predictions[city] = {}
        
        predictions[city]['next_month_predictions_pollution_level'] = next_month_predictions
        predictions[city]['avg_next_year_prediction_pollution_level'] = avg_next_year_prediction
        
        # Repeat for CO, SO2, NOx, O3
        for gas in ['CO', 'SO2', 'NOx', 'O3']:
            X_gas = city_df[['month']].values
            y_gas = city_df[gas].values
            
            poly_gas = PolynomialFeatures(degree=2)
            X_poly_gas = poly_gas.fit_transform(X_gas)
            
            model_gas = LinearRegression()
            model_gas.fit(X_poly_gas, y_gas)
            
            next_month_predictions_gas = model_gas.predict(poly_gas.transform(next_months.reshape(-1, 1)))
            next_month_predictions_gas = np.maximum(next_month_predictions_gas, 0.1)  # Adjust as needed
            
            # Predict next year's pollution level for each gas
            avg_next_year_prediction_gas = max(model_gas.predict(poly_gas.transform([[city_df['month'].max() + 12]]))[0], 0.1)  # Adjust as needed
            
            predictions[city][f'next_month_predictions_{gas}'] = next_month_predictions_gas
            predictions[city][f'avg_next_year_prediction_{gas}'] = avg_next_year_prediction_gas
        
        predictions[city]['next_months'] = next_months
        predictions[city]['next_month_predictions'] = next_month_predictions
        predictions[city]['avg_next_year_prediction'] = avg_next_year_prediction

        # Check if the next year's average pollution level exceeds 90 and send an alert
        if avg_next_year_prediction > 90:
            send_alert(city, avg_next_year_prediction)

        # Print predictions
        print(f"Predicted average pollution level in {city} for next year:")
        print(f"  Pollution Level: {avg_next_year_prediction:.2f}")
        for gas in ['CO', 'SO2', 'NOx', 'O3']:
            avg_gas_prediction = predictions[city][f'avg_next_year_prediction_{gas}']
            print(f"  {gas}: {avg_gas_prediction:.2f}")

    return predictions

# Function to plot future pollution predictions for each gas
def plot_future_pollution_predictions(predictions):
    sns.set(style="darkgrid")
    
    for city, data in predictions.items():
        next_months = data['next_months']
        
        plt.figure(figsize=(12, 6))
        
        for gas in ['pollution_level', 'CO', 'SO2', 'NOx', 'O3']:
            key = f'next_month_predictions_{gas}'
            if key in data:
                sorted_indices = np.argsort(next_months)
                sorted_months = next_months[sorted_indices]
                sorted_predictions = data[key][sorted_indices]
                
                plt.plot(sorted_months, sorted_predictions, marker='o', label=gas)
        
        # Convert sorted_months to month names for xticks
        next_month_names = [calendar.month_name[month] for month in sorted_months]
        
        plt.xticks(sorted_months, next_month_names)
        plt.title(f'Predicted Pollution Levels for Next Month - {city}')
        plt.xlabel('Month')
        plt.ylabel('Predicted Pollution Level')
        plt.legend()
        plt.show()

# Example usage
cities = ['Mumbai', 'Delhi', 'Kolkata', 'Chennai']
plot_city_trends(df, cities)
predictions = predict_future_pollution(df, cities)
plot_future_pollution_predictions(predictions)