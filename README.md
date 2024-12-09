<h1 align="center">ImmoEliza: Machine Learning</h1> <br>
<p align="center">
  <a href="https://becode.org/" target="_blank">BeCode</a> learning project.
</p>
<p align="center">AI & Data Science Bootcamp</p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#preprocess">Preprocess</a></li>
        <li><a href="#model">Model</a></li>
      </ul>
    </li>
    <li> <a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributors">Contributors</a></li>
    <li><a href="#timeline">Timeline</a></li>
  </ol>
</details>

## **About The Project**
This project aim is creating a machine learning model to predict real estate prices in Belgium using Decision Tree Regressor and Random Forest Regression Model. 

The initial database contains 37021 observations and 20 columns. This database is the result of the __[scrapping challenge](https://github.com/MaximSchuermans/immo-eliza/blob/main/data/cleaned_data.csv)__, which involved scraping data from the Belgian real estate website [Immoweb](https://www.immoweb.be/). The database was further enhanced by a second round of data scraping using the same scraper. 

#### Preprocess
<details>
<summary><h2>Data augmentation</h2></summary>
Using open data from data.gov.be, statbel.fgov.be and www.politie.be some new features were added. 
- **Municipality**: Added using the postal code of each observation. 
- **Prosperity index**: This index represents the relative average income of a municipality compared to the national average. It was taken from fiscal statistics for 2022. 
- **Population density**: The population per square kilometer for each municipality, based on data from January 2024.  
- **Crime rate**: Using crime statistics per municipality for 2023 and total population per municipality, the crime rate was calculated per 1,000 inhabitants.
- **Median price of properties by municipality**: The median price for each municipality was calculated using data from 2023 and the first two trimesters of 2024, with a combined median of houses and apartments.  
</details>

<details>
<summary><h2>Imputation of missing values</h2></summary>
- **Boolean columns**: The next columns already contain or were converted to boolean columns: 'furnished', 'open_fire', 'terrace', 'garden', 'swimming_pool', 'disabled_access', 'lift', 'type_of_property'. The missing values were filled with zero, assuming that the observation does not have the feature when the value is missing. 
- **Garden and terrace area**: The missing values for these columns were for observations that do not have a garden or terrace. These missing values were filled with 0. 
- **State of the building and number of facades**: The missing values in these columns were imputed using the mode of each group (type_of_property, municipality) and the global mode for remaining missing values. 
- **Living area**: The missing values in this column were imputed using the median of each group (type_of_property, municipality) and the global median for remaining missing values. 
- **Median price per municipality**: The observations with missing values were dropped. 
</details>

<details>
<summary><h2>Feature combination</h2></summary>
- The columns 'garden' and 'terrace' were combined into a boolean column called 'exterior_space'. 
- The column 'accessible' was created by combining 'disabled_access' and 'lift'.
- The 'state_of_the_building' and 'fully_equipped_kitchen' columns were combined into a new one called 'extra_investment', where higher values represent less work required. 
</details>

<details>
<summary><h2>Remove outliers</h2></summary>
- Outliers from 'price', 'living_area', 'number_of_facades' were removed using the IQR (Interquartile Range) method. 
- In the column "number_of_rooms", when the value was 0 for a house, it was imputed with the median of each group (type_of_property, postal_code). All observations with more than 7 rooms were deleted.  
</details>

<details>
<summary><h2>Filter observation</h2></summary>
- Only postal codes that appear more than 30 times in the dataset were retained.
</details>

<details>
<summary><h2>Drop columns</h2></summary>
- **'municipality', 'region', and 'province'**: These columns were used to create new features and filter data and are no longer needed. 
- **'subtype_of_property'**: A large part of the observations had the same value for 'subtype_of_property' as for 'type_of_property'.
- **'terrace_area' and 'garden_area'**: Initially, these columns were combined into 'ext_area', but the majority of the observations had zero values.  
- **'surface_area_plot_of_land', 'surface_of_the_land'**: 'surface_of_the_land' repeated the values from 'garden_area', and apartments had no 'surface_area_plot_of_land'.   
- **'population_km', 'crime_rate','postal_code','accessible', 'furnished', 'open_fire', 'swimming_pool'**: These columns had low correlation with the target.
- 'type_of_property' and 'number of rooms' have more correlation with living_area than with price.
</details>

Summary of the database after all steps: 
- Shape: 18208, 5
- Columns (all are numerical columns): 'price', 'living_area', 'median_price_per_municipality', 'prosperity_index', 'extra_investment'. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Model
- The model was train using the next features: 'living_area',	'median_price_per_municipality', 'extra_investment' and 'prosperity_index'
- Split: The split for test/train dta was done 20/80
- SQRT transformation to targe ('price'): It is also used for reducing right skewness (high values get compressed and low values become more spread out). Before computing metrics, the predictions and actual values were transformed back to the original scale by squaring them.
- Parameters for Decision Tree Regressor: min_samples_split=15, min_samples_leaf=10, max_leaf_nodes=150, max_depth=20
- Parameters for Random Forest Regressor: n_estimators=150, min_samples_split=100, min_samples_leaf=17, max_leaf_nodes=100, max_depth=100

## **Installation**
1. Clone the repository
2. Install the required libraries by running pip install -r requirements.txt

## **Usage**
1. Run both (preprocess and model) running `main.py`

## **Contributors**
* Jessica Rojas - https://github.com/jessrojasal

## **Timeline**
2 Dic 2024 - project initiated 
9 Dic 2024 - project concluded

<p align="right">(<a href="#readme-top">back to top</a>)</p>
