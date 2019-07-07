# DataAnalysisPy

# DataVizPy

DataAnalysisPy is a simple tool to help the Exploratory Data Analysis directly from Jupyter Notebook and Jupyter Lab. 🚀

Data preparation is a tedious and time-consuming task. In most of the case, we need to connect our data to lots of data sources, create new variables and eliminate unwanted observations. During this crucial step of a data project, a good practice would be to check if all of the wranglings do not distort too much the data. For instance, when we are merging two datasets together, it's important to see if the shape or the distribution of the new data frame roughly looks the same as the original data.

  

Once the dataset is ready, we need to spend a tremendous time understanding and investigating the data. Ignoring this step cannot bring successful results. If our task is to use the dataset to make predictions, then we first need to discover the patterns that the algorithms will extrapolate and generalize.

  

## Motivations

This explanatory step requires lots of coding, to know the basic statistical tests and communicate the results easily. DataVizPy is a tool helping to understand the data both visually and statistically. DataVizPy can be imported in a single line of code, no extra coding required afterward. The different data visualization techniques are accessible via tabs and variables can be selected with a dropdown menu.

  

Another motivation behind DataVizPy is to the possibility to save and archive the results. A data project takes lots of time and in most case, we need to check back and forth the results found some time ago. It wouldn't make any sense to reload the data, re-run the analysis and again draw a conclusion. It is easier to save the results either in a spreadsheet or in PDF, directly to the cloud.

  

Another important perk to save results in a spreadsheet (or CSV) is the end-user of the data probably needs to communicate the results in a different way. It's often the case that the consumer of the data changes the type of graph, or use a custom template. It would be time-consuming to manually copy and paste the results of the computation and the data to make a figure or a statement. To overcome this potential issue, DataVizPy communicates with Google Drive to save the results in a Google Spreadsheet.

  

Last but not least, DataVizPy can be used to make batch computations on a set of pre-defined variables. The user indicates what variables to input, DataVizPy makes the computations and save the results in a spreadsheet.

  

## The features

-   Quick data analysis including the following tests:

-   Continuous variable analysis (Scatterplot)
-   Categorical variable (Chi test, Correspondence Analysis)
-   Mixte between continuous and categorical variables (ANOVA)

-   Data transformation:

-   log
-   standardization/normalization (forthcoming)

-   Filter the data:

-   quartile
-   exclude undesired values or groups

-   Filter the results (forthcoming)
-   Save the results in Google Drive or CSV (forthcoming)
-   Batch computations

  

## How it works

  

Open a Jupyter notebook, or Jupyter lab, and install the library

  

```
!pip install git+https://github.com/thomaspernet/DataAnalysisPy
```

Then load your data. Make sure the data have the right format. For instance, if you want to treat a continuous feature as a categorical feature, you need to convert it to a string.

  

You can use this line of code in a single cell to allow Jupyter to display the output without scrolling down (ie, it increases the output window).

  
```
%%javascript

IPython.OutputArea.auto_scroll_threshold = 9999;
```
  

In this example, I will use my own dataset that I use for one of my paper. The dataset contains 25,404 observations with the following variables:

 |  Variable name               |  Description                                        |
| ---------------------------- | --------------------------------------------------- |
| year                         |   Year:  2002 - 2007                                |
| Post                         |  Take value 1 if year > 2005                        |
| prov2013                     |  Province name                                      |
| citycn                       |  City name Chinese                                  |
| geocode4_corr                |  City code                                          |
| CIC                          |   2 digits industry code                            |
| IndustryName                 |  CIC industry name                                  |
| financial_dep_china          |  Financial dependence industry Chinese Data         |
| SO2_emissions_2005           |  SO2 emission in 2005 city level                    |
| ttoutput                     |  Total output city-industry-year level              |
| tso2                         |  Total SO2 city-industry-year level                 |
| so2_intensity_value_added    | Total SO2 over value-added city-industry-year level |
| tso2_mandate_c               |  Amount of SO2 to reduce city level                 |
| avg_ij_o_city_mandate        |  City effort over output                            |
| d_avg_ij_o_city_mandate      |  dummy City effort over output                      |
| fixed_asset                  |  Total fixed asset city-industry-year level         |
| employment                   |  Total employment city-industry-year level          |
| FE_c_y                       |  pair group city-year                               |
| FE_c_i                       |  pair group city industry                           |
| FE_i_y                       |  pair group industry-year                           |

  

The aim of the paper is to study the reasons to reduce the level of SO2 emitted in China.

  

Now, we are ready to load the dataset into DataVizPy

  
```
QuickDescriptive.PyAnalysis(dataframe=df_final,

 automatic = True,

 Date = 'year',

cdr = False)
```
  

There are three arguments (the other one are details later):

-   data frame: Define the data frame to use
-   automatic: Automatically populates the dropdown menu
-   Date: Define the Date variable (optional)
-   cdr: Configuration object to communicate with Google Drive (optional)

  

DataVizPy is open and looks like this:

  

![](https://drive.google.com/uc?export=view&id=1wGtQxxzFXU3KO1ZM_MMUQlmovG82gXmL)

  

  

There are 10 different tabs with a specific use:

  

-   Quick description
-   Time Serie Plot
-   Low dimensional
-   High Dimensional
-   High/Low dimensional
-   Scatter plot
-   Scatter plot, group 1
-   Scatter plot, group 2
-   Categorical analysis
-   Fixed effect

  

  

### Quick Description

This tab uses `pandas_profiling` library. If you want to know more about this library, please go [here](https://github.com/pandas-profiling/pandas-profiling). You can get the report by clicking on `Run Interact` and/or select `export` to download the report locally. Please, ensure you fill the name case.

  

DataVizPy uses [GoogleDrive-python](https://github.com/thomaspernet/GoogleDrive-python)  library to communicate with Google Drive. If you set up Google Drive authorization and pass the object in `cdr` then you can save the report to Google Drive. Make sure you indicate in which folder to save the report.

  

The report provides detailed statistics for each variable.

  

![](https://drive.google.com/uc?export=view&id=1d47XIIySjjCoQeZFUQ3uGcqYiWHcOCvY](https://drive.google.com/uc?export=view&id=1d47XIIySjjCoQeZFUQ3uGcqYiWHcOCvY)

![](https://drive.google.com/uc?export=view&id=1eqY1fmacCfo5m5okH_OKNdvdymSTrKXM](https://drive.google.com/uc?export=view&id=1eqY1fmacCfo5m5okH_OKNdvdymSTrKXM)

  

  

The bottom of the report shows the correlation matrix, both for Pearson and Spearman.

  

### Time Serie plot

  

The second tab provides a time trend for the relevant continuous variable. You have the choice to stack the plot by groups. Since the figure is based on Plotly, there is no restriction in the number of groups for each categorical variable.

  

Each tab contains `sheedID`, `sheetName` and sometimes `folder`, with two options `move_to_drive` and `move_to_drive_batch`.

  

It gives you the opportunity to save the results in a Google Spreadsheet. For that, you need to create a new spreadsheet, with a sheet name. In the picture below, the `sheetID` is *1PcS_XXXXX* and the results will be paste in the sheet named *test_1*. The output is also available as a seaborn image and saved in the folder `temp_library`

  

![](https://drive.google.com/uc?export=view&id=10rNJcpo6dbNP78bWyWZ_Xir0RNmi_gQY)

  

The output is the following:

  

-   The sum of SO2 by province location

![](https://drive.google.com/uc?export=view&id=1Y3HsabsqFZpxLeKMw2PpW9ySbUoVGEMi)

-   The second tab displays the mean/median and sum. This image will be stored in Google Drive.

  

In Google Spreadsheet, it looks like this:

  

![](https://drive.google.com/uc?export=view&id=1DNjjVI3DP3Y8wu5_rRF89z6EBB3BSniH)

  

If you want to plot another variable and save the results to the same spreadsheet/sheet, the output is pasted few cells down to the previous results.

  

Note that, a batch computation will take all the variables in the dropdown menu, computes the sum and pastes the results in the spreadsheet with the image. No group plot available. If you have dozens of variables in the dropdown menu, I do not recommend to do a batch computation. Instead, you can manually create a dictionary with the preferred variables. I'll explain later how to do it.

  

### Low dimension

  

I define a low dimension categorical variable when the number of groups is less than 10. If above, it is cast down to high dimension. In this tab, we are interested in the distribution of a continuous variable for each group in the categorical variable. You can select to transform the continuous variable in log or not.

  

The output has three tabs:

-   Distribution:

-   it plots the PDF and CDF for each group
- ![](https://drive.google.com/uc?export=view&id=1OHhriHOgFiUAztazp0VLb50BkDWOvQNO)



-   Summary statistics:

-   Provides a quick summary statistics by groups such as number of obs, mean, SD, SE and 95% confidence interval
![](https://drive.google.com/uc?export=view&id=1OK6Cfd2IWx1-EnlK-QJ6xcIUoV6pxd2l)


-   Tukey Results

-   Compute the mean difference of each pair and report the Tukey statistic. If `True` , it means there is enough evidence that the mean is different between these two groups

![](https://drive.google.com/uc?export=view&id=1W0E9StVghhwYSPSHWU0T5NI7UMQogf6v)


  

### High dimension

  

This tab is very similar to the previous one except that we are checking the statistics by groups of high dimension. In our example below, we can see if the output produced by the 39 industries in China are different

  

  

The output has four tabs:

-   Summary statistics:

-   Provides a simple summary statistics, sorted by mean

![](https://drive.google.com/uc?export=view&id=10O2h6VNIPKR83gEFU1zC71HgcLmgk77E)

-   Tukey Results:

-   Show the Tukey statistic pairwise matrix

![](https://drive.google.com/uc?export=view&id=1HeFj5PLMcbD7KaO2mGry5tqbK-w7x7zW)

-   Heatmap

-   Complement the Tukey results by showing the mean difference by groups. Do not pay attention to the color of the cells, there are under construction.

![](https://drive.google.com/uc?export=view&id=1nqUtCiqY81F4uXAes3mizy9tKXM3ksGU)

-   True only

-   Extract only the `True` pairwise difference

![](https://drive.google.com/uc?export=view&id=1t-tkYnpdgtoJHcrWpbjg9V9hLtZIIFG7)

  

It is possible to add an extra level to the data by choosing a low dimension feature from the dropdown menu `var_cat_color`

  

![](https://drive.google.com/uc?export=view&id=1q-J65e4YZfaBHzEyAQphwyZ28vIUA3o3)

  

and the summary statistics displays the different metrics at the new level a disaggregation

  

![](https://drive.google.com/uc?export=view&id=1OXfFHtK2q1rgJKDxsVWVckLAGSNiZC4p)

  

### High/low dimensional

In the next tab, we look at the data from a different angle. We want to highlights if there is a difference between the high dimensional group and the low dimensional group for given continuous variables.

  

This tab is useful if we need to model a difference-in-difference analysis. We aim at visualizing if the trend before and after an event occurred is the same or not between the control group and the treatment group.

  

The output has 4 tabs:

-   rank

-   Plots the rank of the high dimensional group for each subgroup of the low dimensional variable. The rank is computed based on the mean. The slicer can be used to visualize different groups
-   The bottom values are those who have the largest mean, hence have a ranking close to 0.

![](https://drive.google.com/uc?export=view&id=1mJ8z7uP0RFc_J70by9UzE9M1y9auYSFw)

  

If the number of category in the low dimension variable is not too considerable, we can see how large the difference is with the next plot. The x-axis plots the average of the continuous variable and the y-axis plots it's rank. If the dots are widely spread, it implies a large difference both within high dimensional groups and low dimensional groups

  

![](https://drive.google.com/uc?export=view&id=1AL_DX60HZB6FNnWrctbAdCbFmtN4s4uI)

  

-   Summary Statistic

-   Provides the mean/average/median/sum and percentage for each high/low categories

![](https://drive.google.com/uc?export=view&id=1rCtvCcGOexkU3Mx97VQTVI97DZuakFqn)

-   Difference plot

-   Plot the sum/mean/median difference of the continuous variables by high dimension and low dimension if the number of group in the low dimension is equal to two

![](https://drive.google.com/uc?export=view&id=1PnkB96ioFfyxCw7yVdjimz08FfMJOMrT)

  

We can visualize the rank plot with a larger amount of low dimensional variable such as the year.

It makes more sense how some provinces have reduced their emission of SO2, on average.

  

![](https://drive.google.com/uc?export=view&id=1hyWGc1WgQOE3J6rfBIW1M3o4PSrR19d0)

  

The next three tabs will show the relationship between two continuous variables, at different levels.

### Scatterplot

  

The first tab of the scatterplot bundle shows the raw distribution between an X-variable and a Y-variable. It is possible to choose which variable to log-transformed; X, Y or both

  

![](https://drive.google.com/uc?export=view&id=1bRoi86M5_ka6x1A9ZQcvxqMKOEy_N8ML)

  

The computation can be slow if the data frame is large. For now, the plot uses the regular scatter plot function from Plotly. In the future, the scatter plot will rely on the `scatterg` , ie large data frame.

  

![](https://drive.google.com/uc?export=view&id=1bba14i5AFAh8RPa87NJEk-B9m0u97chw)

  

### Scatter plot, group 1

  

The second tab of the scatter plot bundle allows adding one dimension to the scatter plot. It gives the possibility to the user to choose how to summarize the data (mean/median/sum/min/max) for a given low dimensional variable.

  

In the example below, we want to show the relationship between the output and the emission of SO2, aggregated by the location of the province (East, Central or West) using the mean. Note that, you can change the way to aggregate the data for each variable, like mean of X and sum of Y

  

![](https://drive.google.com/uc?export=view&id=1AYb78nrmX0RQ5mp6iGvDbGLkg-uDWnnJ)

  

The output has three tabs:

-   Scatter plot, aggregated:

-   Plot the aggregated using the personalized aggregation method

![](https://drive.google.com/uc?export=view&id=1BpoUpDC9cqOOZFVa3GiHUgFn24UlCibq)

-   Plot scatter:

-   Plot the raw coordinates of X and Y, for each subgroup

![](https://drive.google.com/uc?export=view&id=1Xr-tXp3iOU5H0DWrFTi4CeSMnz3u8LKY)

-   Linear regression

-   Compute the linear relationship between X and Y, for each subgroup **separately**

![](https://drive.google.com/uc?export=view&id=1SRDJOP4aPldqnODVWhFr1cfXM3QFmW5x)

  

  

### Scatter plot, group 2

  

The last technique to visualize the relationship between X and Y is to aggregate the data for one high dimensional variable and color the dots with a low dimensional variable.

  

For instance, we can aggregate the output and emission of SO2 by cities, and color the cities according to their location in China: East, Central or Western.

  

The output has three tabs:

-   Scatter plot, filter

-   Aggregate X and Y variables with the selected group and color the data using the group from `var_col`.

![](https://drive.google.com/uc?export=view&id=1_sC4-GnR1rb6dmZfFzvXH_40OT7SHkxx)

-   Scatter plot

-   Plot the linear relationship between X and Y, using the mean/median and sum for aggregation

![](https://drive.google.com/uc?export=view&id=1ZY9fDkFA3oTzjvdUswbre7tCu3_OhqVv)

-   Scatter plot, color

-   it replicates the first tab, not only aggregate by sum, but also by mean and median
-   This tab is primarily useful to save the images in the cloud.

![](https://drive.google.com/uc?export=view&id=1CWMXMjImyLBSuKxRCFJxetBoSHNukjch)

  

### Categorical Analysis

  

One of the last analysis we can perform on the data is a categorical analysis. In this tab, we are interested in the proportion of rows that fall between two categorical variables. To extract interesting information, we use the chi-square test and correspondence analysis. Both analyses are very correlated since they are based on the residual.

  

As an illustrative example, we can see if there is a change between the number of industries in China by provinces. Say differently, we want to know if some provinces have more industries than others and if yes, which one. If the number of categories is too large, we recommend switching to the next tab, fixed effect

  

The output has a large number of tabs to provide the maximum information:

-   table_count

-   the first table gives the raw count of observations between the rows and the columns
-   In our example, the rows are the industries and the columns are the provinces. The matrix is quite large since there are 29 provinces and 39 industries
-   It is difficult to interpret, so we need to look at the other tabs

![](https://drive.google.com/uc?export=view&id=1Dq1kDweswl271gH_1I5f8c0KFhSFz9GL)

-   table_row_proportion

-   the row proportion table counts the percentage of observations for a given row
-   The darker color indicates a higher proportion

![](https://drive.google.com/uc?export=view&id=1JRZF5SnPDaWTo_ZEPhhrs-sT6xsCEY6R)

-   table_proportion_col

-   The column proportion table is similar in spirit than the row proportion but compute the percentage by column

![](https://drive.google.com/uc?export=view&id=1zrdX7gOAxG45Ntn8rj4-NRl5V9Ujpg-K)

-   pearson_resid

-   The Pearson residual table looks at the association between the rows and columns. It tells which row, for a given column, has a larger proportion than expected
-   The darker values highlight cells with a positive association between the corresponding row and column variables.
-   The lighter values highlight a negative association between the corresponding row and column variables.

![](https://drive.google.com/uc?export=view&id=187aBRITkEWgdI-kbMRSraneRonKOYeD2)

-   contribution

-   The contribution table shows the contribution in % of a given cell to the total Chi-square score is calculated square of the residual divided by the Chi statistics

![](https://drive.google.com/uc?export=view&id=1TVSpEoOLutSNvPz3fvbu4iS0GWCkYUC4)

-   result_test

-   The last tab shows the results of the Chi-square test

![](https://drive.google.com/uc?export=view&id=1yZslLVzwkMg8qRvtpWtqXxQP8WI-o5Mh)

  

### Correspondence Analysis

If both categorical variables have more than three groups, it is possible to run a correspondence analysis. To make a correspondence analysis, you need to select `ca`

  

We can run a CA for the industry and the year. It informs how the industries have evolved over time.

  

The graph plots the relative association between the rows and the columns using the rows as reference. In our example, `year` is the row and `Industry` is the columns. A closer angle from the origin between the row variable and the column indicates a relative positive association. By analogy, an angle close to 180 degrees indicates a relative negative association. An angle of 90 degrees indicates no association.

  

For instance, the year 2007 has relatively more observations relative to Medicine industries and no association at all with Culture, Education and Sport Activity.

  

![](https://drive.google.com/uc?export=view&id=1rwLFN12qjG10IWdyit0BQdO0m7PJtwq7)

  

### Fixed effect

  

The last tab provides a visualization between two high dimensional categorical variables. The correspondence analysis or Chi-square analysis is difficult to interpret when the matrix is very large. To overcome this issue, we can use the fixed effect tab to map the relationship between two categorical variables.

  

Imagine you are interested in the relationship between the distribution of industries for all the cities in the dataset. The matrix is consequent since there are more than 250 cities in the dataset and 39 industries.

  

Group 1 represents the y-axis while group 2 is the x-axis.

  

![](https://drive.google.com/uc?export=view&id=10q_iL-Pi3z3CVEd793-Ab5OkJSGLwv4n)

  

The output has two graphs. The first graph provides a distinct count of group 2 for each group 1. For instance, the city of Shenzhen has 28 industries over 29 possible choices.

  

![](https://drive.google.com/uc?export=view&id=1Z4p-GxH9_k6fnywtLRT_NP12mGMeTkoB)

  

The second graph shows the count distribution. It can be read from left to right, top to bottom.

  

![](https://drive.google.com/uc?export=view&id=11eN4skYLWIthzFfmyPm2I1pM85JEQBVE)

  

First of all, cities with red values have a larger count of observations than the blue one. For instance, industry like Sports Activity is concentrated in very few cities in China (left-right). Sport Activity has more blue values spread across the row than red. You can zoom in to see which cities have, indeed, Sports activities industry. The city of Shenyang has four observations. It means the industry Sports Activity appears for four years in the city of Shenyang. Almost all other cities do not have this industry.

  

![](https://drive.google.com/uc?export=view&id=1s0XlshJvQyHu7MuGTratr6ACilwwTTt2)

By analogy, the industry of non-mineral products looks to be evenly spread across the dataset both in by time and location

  

![](https://drive.google.com/uc?export=view&id=1ZaY5wlLWvrafRVuy0KrX1oJYd_UTWRbJ)

  

A top-down perspective gives us information about the number of group 2 (ie industry) for a given group 1 (ie. city). Values with more red indicate that this particular city has many industries for many years.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MzkyODY1NTgsLTM1OTYwMzQ3MSwyND
E1MzM5NzYsLTM4NTIxMDE3NCw5MjMxMzI3MDIsMzY2MjA5ODNd
fQ==
-->