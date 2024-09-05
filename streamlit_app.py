import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define columns to exclude
EXCLUDE_COLUMNS = [
    'PLAYER', 'EFF', 'Year', 'BPM', 'MAX_VERTICAL_LEAP_y', 'MP.1', 'WS', 'BPM', 'VORP',
    'STANDING_VERTICAL_LEAP_x', 'MAX_VERTICAL_LEAP_x', 'MAX_BENCH_PRESS_x', 'HEIGHT_W_SHOES',
    'STANDING_REACH', 'WEIGHT', 'WINGSPAN', 'STANDING_VERTICAL_LEAP_y', 'MAX_VERTICAL_LEAP_y',
    'MAX_BENCH_PRESS_y','LANE_AGILITY_TIME_x',
       'THREE_QUARTER_SPRINT_x','LANE_AGILITY_TIME_y', 'THREE_QUARTER_SPRINT_y'
]

# Load and prepare the data
@st.cache
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ZSChua/NBA-Stats/main/final_data.csv')
    # Select features by excluding specified columns
    feature_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    X_multi = df[feature_columns]
    Y_target = df['EFF']
    # Impute missing values with the mean of the column
    X_multi = X_multi.fillna(X_multi.mean())
    return X_multi, Y_target, feature_columns

def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    lreg = LinearRegression()
    lreg.fit(X_train, Y_train)
    return lreg

# Load data and train model
X_multi, Y_target, feature_columns = load_data()
lreg = train_model(X_multi, Y_target)

# Streamlit app
def main():
    st.title('NBA Player Efficiency Prediction')
    st.text('Fill in the stats below and hit predict to find the result.\nEnter all percentages in decimal value (30% = 0.3).\nFeel free to leave values empty. They will be filled with the average value.')

    name_map = {
        'FG%':'Field Goal Percentage',
        '3P%': 'Three Point Percentage',
        'FT%': 'Free throw Percentage',
        'PTS.1': 'Points Per Game',
        'TRB.1': 'Rebounds Per Game',
        'AST.1': 'Assists Per Game',
        'WS/48': 'WS/48 Win Shares Per 48 Mintues',
        'BODY_FAT': 'Body Fat Percentage',
        'HEIGHT_WO_SHOES': 'Height in without shoes in cm'

    }

    feature_values = []
    for col in feature_columns:
        value = st.text_input(f'{name_map.get(col)}', value='Unknown')
        if value == 'Unknown':
            value = X_multi[col].mean()  # Use the average value from the DataFrame
        else:
            try:
                value = float(value)  # Convert to float
            except ValueError:
                st.error(f'Invalid input for {col}. Please enter a numeric value.')
                return  # Exit the function if the input is invalid
        feature_values.append(value)

    if st.button('Predict'):
        # Prepare input data
        input_features = [feature_values]
        prediction = lreg.predict(input_features)
        st.write(f'Predicted Efficiency: {prediction[0]}')

if __name__ == "__main__":
    main()
