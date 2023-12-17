# DataTalksClub Machine Learning Zoomcamp Capstone 1 Project: Predict if Today is Raining or Drizzle

## Problem Description
Global Summary of the Day is one of the famous dataset in the BigQuery public data. This public dataset was created by National Oceanic and Atmospheric Administrator (NOAA) and it consists of daily weather elements collected from over 9000 stations in different region of country. This data includes mean value of daily temperatures, sea level pressure, wind speed, precipation and many more metric. The data also includes the column `fog, rain_drizzle, snow_ice_pellets, hail, thunder, tornado_funnel_cloud` which represent the occurence of event related to weather that reported during the day.

In this project, we will use this dataset to predict if the current day will be raining or not. We will use the weather performance in the **last 7 days** from the GSOD data to predict if there is a **sign of rain or drizzle** during the day (the column `rain_drizzle`). Since the dataset is in form of daily performance, it will be needed to perform a feature-engineering to make the last 7 days metric is available as a feature during the training and inferencing. (eg: day-1, day-2, day-n)

## Problem Scope

The machine learning model will use the data only from station id **'967450'**, which is the **JAKARTA/OBSERVATORY - INDONESIA** Station. The data that will be used is data started from **2013** to **2023**. The selected features for this problem are:

Identifier:
- date: represent the date of the current day

Feature:
- year: Represent the year of the current day
- mo: Represent the month of the current day
- da: Represent the day of the current day
- avg_temp: Average temperature that recorded during the day. The metric is in Fanrenheit. Missing = 9999.9
- avg_dew_point: Dew point is a factor in the heat index. It's used to measure the water vapor content of natural gas. Using the average value. Missing = 9999.9
- avg_sea_level_point: Sea level is the average height of the ocean's surface that recorded during the day. Using the average value Missing = 9999.9
- avg_wind_speed: Mean wind speed for the day in knots to tenths. Missing = 999.9
- total_precipitation: Total precipitation (rain and/or melted snow) reported during the day in inches and hundredths; will usually not end with the midnight observation--i.e., may include latter part of previous day. .00 indicates no measurable precipitation (includes a trace). Missing = 99.99

  > Note: Many stations do not report '0' on days with no precipitation--therefore, '99.99' will often appear on these days. Also, for example, a station may only report a 6-hour amount for the period during which rain fell. See Flag field for source of data

- flag_precipitation: 
  - A = 1 report of 6-hour precipitation amount
  - B = Summation of 2 reports of 6-hour precipitation amount
  - C = Summation of 3 reports of 6-hour precipitation amount
  - D = Summation of 4 reports of 6-hour precipitation amount 
  - E = 1 report of 12-hour precipitation amount
  - F = Summation of 2 reports of 12-hour precipitation amount
  - G = 1 report of 24-hour precipitation amount
  - H = Station reported '0' as the amount for the day (eg, from 6-hour reports), but also reported at least one occurrence of precipitation in hourly observations--this could indicate a trace occurred, but should be considered as incomplete data for the day.
  - I = Station did not report any precip data for the day and did not report any occurrences of precipitation in its hourly observations--it's still possible that precip occurred but was not reported

Engineered / Augmented feature:
- <metric_name>_prev_<n>_day: The reported value of the weather metric in the previous n day. For example, `avg_temp_prev_4_day` is the value of average temperature that reported in previous 4 days. So, let say today is 2023-12-14 then `avg_temp_prev_4_day` is average temperature that recorded in 2023-12-10.
 
Target: 
- rain_drizzle: Indicators (1 = yes, 0 = no/not reported) for the occurrence during the day

Here is the query of the prepared data that we'll use in this capstone project:

```sql

WITH src AS (
  SELECT
    CASE
      WHEN date is not null 
      THEN date
      ELSE CAST(year || '-' || mo || '-' || da AS DATE)
    END AS date,
    year,
    mo,
    da,
    temp as avg_temp,
    dewp as avg_dew_point,
    slp as avg_sea_level_point,
    wdsp as avg_wind_speed,
    prcp as total_precipitation,
    flag_prcp as flag_precipitation,
    rain_drizzle,
  FROM `bigquery-public-data.noaa_gsod.gsod*`
  WHERE _TABLE_SUFFIX IN (
      '2013',
      '2014',
      '2015',
      '2016',
      '2017',
      '2018',
      '2019',
      '2020',
      '2021',
      '2022',
      '2023'
    )
    AND stn = '967450' #JAKARTA/OBSERVATORY - INDONESIA Station
)

, fill_missing_date AS (
  SELECT
    date_fill as date,
    FORMAT_DATE('%Y', date_fill) as year,
    FORMAT_DATE('%m', date_fill) as mo,
    FORMAT_DATE('%d', date_fill) as da,
    b.* EXCEPT(`date`, year, mo, da)
  FROM UNNEST(GENERATE_DATE_ARRAY('2013-01-01', '2023-12-13')) as date_fill
  LEFT JOIN src b
    ON b.date = date_fill
)

, lagging_window AS (
  SELECT
    date,
    year,
    mo,
    da,
    LAG(avg_temp, 7) OVER(date_window) as avg_temp_prev_7_day,
    LAG(avg_temp, 6) OVER(date_window) as avg_temp_prev_6_day,
    LAG(avg_temp, 5) OVER(date_window) as avg_temp_prev_5_day,
    LAG(avg_temp, 4) OVER(date_window) as avg_temp_prev_4_day,
    LAG(avg_temp, 3) OVER(date_window) as avg_temp_prev_3_day,
    LAG(avg_temp, 2) OVER(date_window) as avg_temp_prev_2_day,
    LAG(avg_temp, 1) OVER(date_window) as avg_temp_prev_1_day,
    LAG(avg_dew_point, 7) OVER(date_window) as avg_dew_point_prev_7_day,
    LAG(avg_dew_point, 6) OVER(date_window) as avg_dew_point_prev_6_day,
    LAG(avg_dew_point, 5) OVER(date_window) as avg_dew_point_prev_5_day,
    LAG(avg_dew_point, 4) OVER(date_window) as avg_dew_point_prev_4_day,
    LAG(avg_dew_point, 3) OVER(date_window) as avg_dew_point_prev_3_day,
    LAG(avg_dew_point, 2) OVER(date_window) as avg_dew_point_prev_2_day,
    LAG(avg_dew_point, 1) OVER(date_window) as avg_dew_point_prev_1_day,
    LAG(avg_sea_level_point, 7) OVER(date_window) as avg_sea_level_point_prev_7_day,
    LAG(avg_sea_level_point, 6) OVER(date_window) as avg_sea_level_point_prev_6_day,
    LAG(avg_sea_level_point, 5) OVER(date_window) as avg_sea_level_point_prev_5_day,
    LAG(avg_sea_level_point, 4) OVER(date_window) as avg_sea_level_point_prev_4_day,
    LAG(avg_sea_level_point, 3) OVER(date_window) as avg_sea_level_point_prev_3_day,
    LAG(avg_sea_level_point, 2) OVER(date_window) as avg_sea_level_point_prev_2_day,
    LAG(avg_sea_level_point, 1) OVER(date_window) as avg_sea_level_point_prev_1_day,
    LAG(avg_wind_speed, 7) OVER(date_window) as avg_wind_speed_prev_7_day,
    LAG(avg_wind_speed, 6) OVER(date_window) as avg_wind_speed_prev_6_day,
    LAG(avg_wind_speed, 5) OVER(date_window) as avg_wind_speed_prev_5_day,
    LAG(avg_wind_speed, 4) OVER(date_window) as avg_wind_speed_prev_4_day,
    LAG(avg_wind_speed, 3) OVER(date_window) as avg_wind_speed_prev_3_day,
    LAG(avg_wind_speed, 2) OVER(date_window) as avg_wind_speed_prev_2_day,
    LAG(avg_wind_speed, 1) OVER(date_window) as avg_wind_speed_prev_1_day,
    LAG(total_precipitation, 7) OVER(date_window) as total_precipitation_prev_7_day,
    LAG(total_precipitation, 6) OVER(date_window) as total_precipitation_prev_6_day,
    LAG(total_precipitation, 5) OVER(date_window) as total_precipitation_prev_5_day,
    LAG(total_precipitation, 4) OVER(date_window) as total_precipitation_prev_4_day,
    LAG(total_precipitation, 3) OVER(date_window) as total_precipitation_prev_3_day,
    LAG(total_precipitation, 2) OVER(date_window) as total_precipitation_prev_2_day,
    LAG(total_precipitation, 1) OVER(date_window) as total_precipitation_prev_1_day,
    LAG(flag_precipitation, 7) OVER(date_window) as flag_precipitation_prev_7_day,
    LAG(flag_precipitation, 6) OVER(date_window) as flag_precipitation_prev_6_day,
    LAG(flag_precipitation, 5) OVER(date_window) as flag_precipitation_prev_5_day,
    LAG(flag_precipitation, 4) OVER(date_window) as flag_precipitation_prev_4_day,
    LAG(flag_precipitation, 3) OVER(date_window) as flag_precipitation_prev_3_day,
    LAG(flag_precipitation, 2) OVER(date_window) as flag_precipitation_prev_2_day,
    LAG(flag_precipitation, 1) OVER(date_window) as flag_precipitation_prev_1_day,
    rain_drizzle
  FROM fill_missing_date
  WINDOW date_window AS (
    ORDER BY date
  )
)

SELECT *
FROM lagging_window
ORDER BY date

```

To make the scope of the problem is building the machine learning model instead of data-engineering, the data will be prepared and stored alongside in this repository on the directory `data/gsod_jakarta_prepared.csv`.

## Exploratory Data Analysis