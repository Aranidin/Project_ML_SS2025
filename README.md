# Project_ML_SS2025
Until 16.06: Do preprocessing

- Work with Text (reviews) and Tabular data and Spatial
  - Listings
  - Calendar
  - Reviews
- Text data (Reviews):
  - Transformers model
- Calendar:
  - Columns:
   - Date available
   - Price
   - (Maybe) Add data about holidays in the area and about the holidays in the other EU countries
- Listings (tabular, spatial):
  - Work with summary .csv file (not the zipped one)
  - Min_nights
  - Max_nights
- Notebooks:
  - 3 x preprocessing
  - 3 x model definitions for different data type
  - 1 Visualization
 
- Spatial (Roo)
- Calendar (Bendix)
- Text (Dinara)

![timeline](timeline.png)


## Data Preprocessing

### Deleted Columns
The columns 'calendar_updated', 'host_neighbourhood', 'host_about', 'neighbourhood', 'neighborhood_overview', 'host_location`
are deleted, because of a large amout of missing values, which are hard to impute

### Beds and Bedroom
All Rows where both are missing are deleted, whith is a rather small amount (~2 % of the original rows).
In rows, where only one is missing, the other value is imputed based on the distribution of values of the other value or the respective value.

### Decisions

- Host profile pic? (yes, no)
- description?
- first review date ?
- has_availability ?
- host_acceptance ?
- host identity verified ?
- host is superhost ?
- host listings count ?
  - do both together and delete rest
- host name ?
- host response rate ?
- host response time ?
- host thumbnail url
- host since
- host total listings count
- host verifications
- last review
- licence
- picture_url
- price
- rewiew-stuff


## Help

- how to analyze Bathroom(types) -> need to fix splitting of bathroom_text
- check bed imputation 

