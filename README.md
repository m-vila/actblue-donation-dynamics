# Data Analyst Technical Assessment

## Project Overview

This project analyzes ActBlue's Federal Election Commission (FEC) filings from February to April 2020, providing insights into donation patterns during a critical period of the 2020 election cycle. The analysis explores how political events, geographic factors, and donor behavior influenced ActBlue's fundraising effectiveness.

## Dataset Description

The analysis uses two primary datasets:

 1. ActBlue's FEC Filing Sample (Feb-Apr 2020) - Contains detailed information about individual contributions processed through ActBlue
 2. FEC Committee Data for the 2020 cycle - Provides information about political committees receiving funds

The project follows a structured data analysis approach:

 1. Data Understanding & Hygiene

* Conducted comprehensive data quality assessment
* Addressed missing values, inconsistencies, and format issues
* Standardized field formats (dates, states, ZIP codes)
* Performed targeted cleaning of city names, employer/occupation fields

 2. Data Transformation & Integration

* Joined datasets on committee identifiers
* Created derived fields for time-based, geographic, and contribution-specific analysis
* Separated contribution types (earmarked vs. direct-to-ActBlue)
* Engineered features to capture political events, regional patterns, and donor behavior

 3. Exploratory Data Analysis

* Analyzed the impact of key political events on donation patterns
* Examined geographic distribution of contributions
* Investigated donor behavior patterns and committee-specific trends

 4. Data Story Development

* Synthesized findings into actionable insights
* Analyzed what drives ActBlue's fundraising momentum during political inflection points

 5. Building the Dashboard
    
 * Key metrics to build the dashboard
 * Extracting the final dataset to use for the dashboard

To visualize the dashboard, click [the following link](https://public.tableau.com/views/ActBluePulseDonationDynamicsDashboard/Dashboard1?:language=es-ES&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

**Future Work**

Potential extensions to this analysis could include:

* Predictive modeling of contribution patterns based on political events
* Donor segmentation analysis to identify high-value supporter groups
* Comparison across multiple election cycles
* Geographic optimization strategy for targeted fundraising campaigns
