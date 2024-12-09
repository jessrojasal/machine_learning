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

## Preprocess
<details>
<summary><h4>Data augmentation</h4></summary>
Using open data from data.gov.be, statbel.fgov.be and www.politie.be some new features were added. 
<ul>
  <li><strong>Municipality:</strong> Added using the postal code of each observation.</li>
  <li><strong>Prosperity index:</strong> This index represents the relative average income of a municipality compared to the national average. It was taken from fiscal statistics for 2022.</li>
  <li><strong>Population density:</strong> The population per square kilometer for each municipality, based on data from January 2024.</li>
  <li><strong>Crime rate:</strong> Using crime statistics per municipality for 2023 and total population per municipality, the crime rate was calculated per 1,000 inhabitants.</li>
  <li><strong>Median price of properties by municipality:</strong> The median price for each municipality was calculated using data from 2023 and the first two trimesters of 2024, with a combined median of houses and apartments.</li>
</ul>
</details>

<details>
<summary><h4>Imputation of missing values</h4></summary>
<ul>
  <li><strong>Boolean columns:</strong> The next columns already contain or were converted to boolean columns: 'furnished', 'open_fire', 'terrace', 'garden', 'swimming_pool', 'disabled_access', 'lift', 'type_of_property'. The missing values were filled with zero, assuming that the observation does not have the feature when the value is missing.</li>
  <li><strong>Garden and terrace area:</strong> The missing values for these columns were for observations that do not have a garden or terrace. These missing values were filled with 0.</li>
  <li><strong>State of the building and number of facades:</strong> The missing values in these columns were imputed using the mode of each group (type_of_property, municipality) and the global mode for remaining missing values.</li>
  <li><strong>Living area:</strong> The missing values in this column were imputed using the median of each group (type_of_property, municipality) and the global median for remaining missing values.</li>
  <li><strong>Median price per municipality:</strong> The observations with missing values were dropped.</li>
</ul>
</details>

<details>
<summary><h4>Feature combination</h4></summary>
<ul>
  <li><strong>Exterior space:</strong> The columns 'garden' and 'terrace' were combined into a boolean column called 'exterior_space'.</li>
  <li><strong>Accessible:</strong> The column 'accessible' was created by combining 'disabled_access' and 'lift'.</li>
  <li><strong>Extra investment:</strong> The 'state_of_the_building' and 'fully_equipped_kitchen' columns were combined into a new one called 'extra_investment', where higher values represent less work required.</li>
</ul>
</details>

<details>
<summary><h4>Remove outliers</h4></summary>
<ul>
  <li><strong>Outliers:</strong> Outliers from 'price', 'living_area', 'number_of_facades' were removed using the IQR (Interquartile Range) method.</li>
  <li><strong>Number of rooms:</strong> In the column "number_of_rooms", when the value was 0 for a house, it was imputed with the median of each group (type_of_property, postal_code). All observations with more than 7 rooms were deleted.</li>
</ul>
</details>

<details>
<summary><h4>Filter observation</h4></summary>
<ul>
  <li><strong>Postal codes:</strong> Only postal codes that appear more than 30 times in the dataset were retained.</li>
</ul>
</details>

<details>
<summary><h4>Drop columns</h4></summary>
<ul>
  <li><strong>Municipality, region, and province:</strong> These columns were used to create new features and filter data and are no longer needed.</li>
  <li><strong>Subtype of property:</strong> A large part of the observations had the same value for 'subtype_of_property' as for 'type_of_property'.</li>
  <li><strong>Terrace area and garden area:</strong> Initially, these columns were combined into 'ext_area', but the majority of the observations had zero values.</li>
  <li><strong>Surface area plot of land, surface of the land:</strong> 'Surface_of_the_land' repeated the values from 'garden_area', and apartments had no 'surface_area_plot_of_land'.</li>
  <li><strong>Population per km, crime rate, postal code, accessible, furnished, open fire, swimming pool:</strong> These columns had low correlation with the target.</li>
  <li><strong>Type of property and number of rooms:</strong> These columns have more correlation with 'living_area' than with 'price'.</li>
</ul>
</details>

Summary of the database after all steps: 
* Shape: 18208, 5
* Columns (all are numerical columns): 'price', 'living_area', 'median_price_per_municipality', 'prosperity_index', 'extra_investment'. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model
<details>
<summary><h4>Decision Tree Regressor and Random Forest Regressor</h4></summary>
<ul>
  <li><strong>Features used for training:</strong> The model was trained using the following features: 'living_area', 'median_price_per_municipality', 'extra_investment', and 'prosperity_index'.</li>
  <li><strong>Split:</strong> The data was split into train/test sets with an 80/20 ratio.</li>
  <li><strong>SQRT transformation to target ('price'):</strong> This transformation was applied to reduce right skewness (high values get compressed and low values become more spread out). Before computing metrics, the predictions and actual values were transformed back to the original scale by squaring them.</li>
  <li><strong>Parameters for Decision Tree Regressor:</strong> min_samples_split=15, min_samples_leaf=10, max_leaf_nodes=150, max_depth=20.</li>
  <li><strong>Parameters for Random Forest Regressor:</strong> n_estimators=150, min_samples_split=100, min_samples_leaf=17, max_leaf_nodes=100, max_depth=100.</li>
</ul>
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
