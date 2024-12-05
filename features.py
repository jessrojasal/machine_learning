import pandas as pd

# Load the CSV file
df_selected = pd.read_csv('data/real_estate_belguim.csv', sep=',')

# Create the DataFrame with selected columns
df = pd.DataFrame({
    'price': df_selected['price'],
    'longitude': df_selected['longitude'],
    'latitude': df_selected['latitude'],
    #'region': df_selected['region'],
    'province': df_selected['province'],
    'crime_rate': df_selected['crime_rate'],
    'population_km': df_selected['population_km'],
    'prosperity_index': df_selected['prosperity_index'],
    'type_of_property': df_selected['type_of_property'],
    'state_of_the_building': df_selected['state_of_the_building'],
    'number_of_rooms': df_selected['number_of_rooms'],
    'living_area': df_selected['living_area'],
    'fully_equipped_kitchen': df_selected['fully_equipped_kitchen'],
    'furnished': df_selected['furnished'],
    'open_fire': df_selected['open_fire'],
    #'surface_area_plot_of_land': df_selected['surface_area_plot_of_land'],
    'number_of_facades': df_selected['number_of_facades'],
    'swimming_pool': df_selected['swimming_pool'],
    'ext_area': df_selected['ext_area'],
    'postal_code': df_selected['postal_code'],
    'municipality': df_selected['municipality'],
    'subtype_of_property': df_selected['subtype_of_property'],

})

# Filter the 'prosperity_index' column
prosperity_counts = df['municipality'].value_counts()
valid_prosperity_values = prosperity_counts[prosperity_counts > 10].index
df = df[df['municipality'].isin(valid_prosperity_values)]

# Filter the 'prosperity_index' column
#df = df[df['type_of_property'] == 'apartment']

# Filter the 'prosperity_index' column
#df = df[df['type_of_property'] == 'apartment'] 0.504003 / 0.695142

# Save the new DataFrame to a CSV file
df.to_csv('data/real_estate_belguim_features.csv', index=False)
