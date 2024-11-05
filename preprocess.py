import pandas as pd

def preprocess_and_split_data(file_path, training_path, test_path):
    # Read the data file, skipping every even-numbered line
    data = pd.read_csv(file_path, skiprows=lambda x: x % 2 == 1, header=0, encoding='utf-8-sig')

    # Strip whitespace from column names to ensure clean access
    data.columns = data.columns.str.strip()

    # Rename columns explicitly to ensure they're accessible
    data.columns = [
        'season', 'week', 'time_et', 'neutral', 'away', 'home', 'score_away', 'score_home', 'ppg_away', 
        'papg_away', 'ppg_home', 'papg_home', 'first_downs_away', 'first_downs_away_allowed', 
        'first_downs_home', 'first_downs_home_allowed', 'first_downs_from_passing_away', 
        'first_downs_from_passing_away_allowed', 'first_downs_from_passing_home', 
        'first_downs_from_passing_home_allowed', 'first_downs_from_rushing_away', 
        'first_downs_from_rushing_away_allowed', 'first_downs_from_rushing_home', 
        'first_downs_from_rushing_home_allowed', 'first_downs_from_penalty_away', 
        'first_downs_from_penalty_away_allowed', 'first_downs_from_penalty_home', 
        'first_downs_from_penalty_home_allowed', 'third_down_comp_away', 
        'third_down_comp_away_allowed', 'third_down_att_away', 'third_down_att_away_allowed', 
        'third_down_comp_home', 'third_down_comp_home_allowed', 'third_down_att_home', 
        'third_down_att_home_allowed', 'fourth_down_comp_away', 'fourth_down_comp_away_allowed', 
        'fourth_down_att_away', 'fourth_down_att_away_allowed', 'fourth_down_comp_home', 
        'fourth_down_comp_home_allowed', 'fourth_down_att_home', 'fourth_down_att_home_allowed', 
        'plays_away', 'plays_away_allowed', 'plays_home', 'plays_home_allowed', 'drives_away', 
        'drives_away_allowed', 'drives_home', 'drives_home_allowed', 'yards_away', 'yards_away_allowed', 
        'yards_home', 'yards_home_allowed', 'pass_comp_away', 'pass_comp_away_allowed', 'pass_att_away', 
        'pass_att_away_allowed', 'pass_yards_away', 'pass_yards_away_allowed', 'pass_comp_home', 
        'pass_comp_home_allowed', 'pass_att_home', 'pass_att_home_allowed', 'pass_yards_home', 
        'pass_yards_home_allowed', 'sacks_num_away', 'sacks_num_away_allowed', 'sacks_yards_away', 
        'sacks_yards_away_allowed', 'sacks_num_home', 'sacks_num_home_allowed', 'sacks_yards_home', 
        'sacks_yards_home_allowed', 'rush_att_away', 'rush_att_away_allowed', 'rush_yards_away', 
        'rush_yards_away_allowed', 'rush_att_home', 'rush_att_home_allowed', 'rush_yards_home', 
        'rush_yards_home_allowed', 'pen_num_away', 'pen_yards_away', 'pen_num_home', 'pen_yards_home', 
        'fumbles_away', 'fumbles_away_recovered', 'fumbles_home', 'fumbles_home_recovered', 
        'interceptions_away', 'interceptions_away_received', 'interceptions_home', 
        'interceptions_home_received', 'possession_away', 'possession_home'
    ]

    # Filter the data into training and test sets based on the 'season' column
    training_data = data[data['season'] <= 2022].drop(columns=['season', 'week', 'time_et', 'neutral', 'away', 'home'])
    test_data = data[data['season'] == 2023].drop(columns=['season', 'week', 'time_et', 'neutral', 'away', 'home'])

    # Save the processed training and test data to new files
    training_data.to_csv(training_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Processed TRAINING data saved to {training_path}")
    print(f"Processed TEST data saved to {test_path}")

# Example usage:
file_path = 'TRAINABLE_teamdata.csv'  # Modify with your file path
training_path = 'TRAINING_2.csv'
test_path = 'TEST_2.csv'
preprocess_and_split_data(file_path, training_path, test_path)
