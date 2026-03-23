### *Turning raw data into decisions — one notebook at a time*
 
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-Advanced-F29111?style=for-the-badge&logo=postgresql&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Expert-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
 
</div>
 
---
 
## 🧠 About This Repository
 
This is a living portfolio of end-to-end data science projects — from scraping raw data off the web to building machine learning models, engineering features, and producing publication-quality visualizations.
 
Every notebook tells a complete story: **data acquisition → cleaning → exploration → feature engineering → modeling → insight.**
 
> 💡 *The long-term goal of this repo is a fully automated data analysis pipeline — each project is a building block toward a one-click insight engine.*
 
---
 
## 🗂️ Project Index
 
| # | Project | Domain | Key Skills |
|---|---------|--------|------------|
| 🚨 | [**911 Emergency Calls**](#-911-emergency-calls-analysis) | Public Safety | EDA, Time Series, Geospatial |
| 🦠 | [**COVID-19 Analysis**](#-covid-19-analysis--web-scraping) | Epidemiology | Web Scraping, BeautifulSoup, Pandas |
| 📸 | [**Instagram Analytics**](#-instagram-analytics-suite) | Social Media | Scraping, NLP, Visualization |
| 🎮 | [**Video Game Sales**](#-video-game-sales) | Gaming / Business | Feature Engineering, Regression |
| 🌍 | [**World Happiness**](#-world-happiness-report) | Social Science | Correlation, Clustering |
| 🎬 | [**YouTube Trending**](#-youtube-trending-videos) | Media / NLP | Text Processing, NLP |
| 🌿 | [**Vitamins & Food Scraping**](#-vitamins--food-scraping) | Health / Nutrition | Web Scraping, Data Cleaning |
| 👁️ | [**People Detection**](#-people-detection-cv) | Computer Vision | OpenCV, YOLO/CV Models |
| 🚔 | [**Police Shootings**](#-police-shootings-analysis) | Public Policy | Statistics, EDA |
| 💣 | [**World Terrorism**](#-world-terrorism) | Global Security | Geospatial, Heatmaps |
| 📣 | [**Sponsored Marketing**](#-sponsored-marketing) | Marketing | Funnel Analysis, ROI |
| 🤖 | [**DS Automation Engine**](#-data-science-automation-engine) | Engineering | Automation, Pipeline |
 
---
 
## 📂 Project Deep Dives
 
### 🚨 911 Emergency Calls Analysis
> Montgomery County, PA — 100K+ emergency call records
 
**What I did:**
- Parsed timestamps to extract hour/day/month features from raw datetime strings
- Created reason-category features via string splitting and `.map()` lambda functions
- Built heatmaps correlating day-of-week × hour-of-day call volumes
- Identified seasonal spikes in EMS vs. Traffic vs. Fire emergencies
 
```python
# Feature Engineering: extracting temporal features
df['Hour']   = df['timeStamp'].apply(lambda t: t.hour)
df['Month']  = df['timeStamp'].apply(lambda t: t.month)
df['Day of Week'] = df['timeStamp'].apply(lambda t: t.dayofweek)
df['Reason'] = df['title'].apply(lambda t: t.split(':')[0])
 
# Cross-tab heatmap: hour vs. day
dayHour = df.groupby(by=['Day of Week', 'Hour'])['Reason'].count().unstack()
sns.heatmap(dayHour, cmap='viridis')
```
 
---
 
### 🦠 COVID-19 Analysis & Web Scraping
> Real-time pandemic data scraped and analyzed
 
**What I did:**
- Used `BeautifulSoup` to scrape live COVID stats from web tables
- Cleaned inconsistent number formats (e.g., `"1,234"` → `1234`) with regex
- Computed rolling 7-day averages and growth rate features
- Visualized country-by-country progression with log-scale plots
 
```python
# Web scraping with BeautifulSoup
from bs4 import BeautifulSoup
import requests
 
page  = requests.get(url)
soup  = BeautifulSoup(page.content, 'html.parser')
table = soup.find('table', {'id': 'main_table_countries_today'})
rows  = table.find_all('tr')
 
# Cleaning: strip commas and cast to int
df['TotalCases'] = df['TotalCases'].str.replace(',', '').astype(int)
 
# Feature: 7-day rolling growth rate
df['growth_rate'] = df['TotalCases'].pct_change(periods=7).round(4)
```
 
---
 
### 📸 Instagram Analytics Suite
> Three-part project: Scraping → Manipulation → Visualization
 
**What I did:**
- Scraped follower counts, post metadata, and engagement stats via API/scraper
- Engineered an **engagement rate** feature: `(likes + comments) / followers`
- NLP on captions: hashtag frequency analysis, sentiment scoring
- Built interactive follower-growth visualizations
 
```python
# Feature Engineering: engagement rate
df['engagement_rate'] = (df['likes'] + df['comments']) / df['followers']
 
# NLP: hashtag extraction
import re
df['hashtags'] = df['caption'].apply(lambda x: re.findall(r'#\w+', str(x)))
df['hashtag_count'] = df['hashtags'].apply(len)
 
# Top hashtags by frequency
from collections import Counter
all_tags = [tag for tags in df['hashtags'] for tag in tags]
top_tags = Counter(all_tags).most_common(20)
```
 
---
 
### 🎮 Video Game Sales
> Global sales data across platforms, genres, and publishers
 
**What I did:**
- Handled missing values with median imputation by genre group
- Engineered `total_sales_ratio` and `regional_dominance` features
- Ran correlation analysis between critic scores and global sales
- Decision tree for sales tier classification (High / Mid / Low)
 
```python
# Grouped median imputation
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
df['User_Score'] = df.groupby('Genre')['User_Score'] \
                     .transform(lambda x: x.fillna(x.median()))
 
# Feature: NA sales as % of global
df['NA_dominance'] = df['NA_Sales'] / df['Global_Sales']
 
# SQL-style query with pandas
top_publishers = (df.groupby('Publisher')['Global_Sales']
                    .sum()
                    .sort_values(ascending=False)
                    .head(10))
```
 
---
 
### 🌍 World Happiness Report
> UN happiness index vs. economic and social indicators
 
**What I did:**
- Correlation matrix across 10+ happiness indicators
- K-Means clustering to group countries by happiness profile
- PCA dimensionality reduction for 2D cluster visualization
- Identified the top predictors of happiness with feature importance
 
```python
# K-Means Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
 
features = ['GDP per capita', 'Social support', 'Healthy life expectancy',
            'Freedom', 'Generosity', 'Perceptions of corruption']
 
X = StandardScaler().fit_transform(df[features])
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
 
# PCA for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X)
```
 
---
 
### 🎬 YouTube Trending Videos
> Trending data across multiple countries
 
**What I did:**
- NLP on video titles: TF-IDF vectorization, common n-gram extraction
- Feature engineering: `days_to_trend` from publish date to trending date
- Category-level aggregations (Music vs. Gaming vs. News performance)
- Sentiment analysis on titles with `TextBlob`
 
```python
# Days to trend feature
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['days_to_trend'] = (df['trending_date'] - df['publish_time'].dt.normalize()).dt.days
 
# TF-IDF on titles
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X = tfidf.fit_transform(df['title'])
```
 
---
 
### 🤖 Data Science Automation Engine
> The flagship project — an automated EDA & reporting pipeline
 
**What I did:**
- Built a reusable class that auto-detects column types (numeric, categorical, datetime)
- Auto-generates distribution plots, missing value heatmaps, and correlation matrices
- Outputs a full HTML report from any DataFrame in one function call
- Designed to be dataset-agnostic — the foundation for a one-click analysis tool
 
```python
class AutoEDA:
    def __init__(self, df):
        self.df  = df
        self.num_cols  = df.select_dtypes(include='number').columns.tolist()
        self.cat_cols  = df.select_dtypes(include='object').columns.tolist()
        self.date_cols = df.select_dtypes(include='datetime').columns.tolist()
 
    def missing_report(self):
        missing = self.df.isnull().sum()
        pct     = (missing / len(self.df) * 100).round(2)
        return pd.DataFrame({'Missing': missing, 'Pct': pct}) \
                 .query('Missing > 0') \
                 .sort_values('Pct', ascending=False)
 
    def run_full_report(self):
        self.missing_report()
        self.plot_distributions()
        self.correlation_heatmap()
        self.categorical_summary()
```
 
---
 

---
 
## 🧩 Core Competencies
 
```
📊 Exploratory Data Analysis    ████████████████████ Expert
🔧 Feature Engineering          ███████████████████░ Advanced
🧹 Data Cleaning & Wrangling    ████████████████████ Expert
🤖 Machine Learning             ████████████████░░░░ Advanced
🕸️  Web Scraping                ████████████████████ Expert
📈 Data Visualization           ███████████████████░ Advanced
💬 NLP & Text Analysis          ████████████░░░░░░░░ Intermediate
👁️  Computer Vision             ████████████░░░░░░░░ Intermediate
⚙️  Pipeline Automation         ███████████████░░░░░ Advanced
```
 
---
 
## 🔍 SQL Showcase
 
Beyond Python, many of the analytical patterns here translate directly into production SQL — here's a taste of the kind of queries that power these analyses:
 
```sql
-- Feature: engagement rate by category, with rolling 7-day average
WITH base AS (
    SELECT
        video_id,
        category,
        trending_date,
        views,
        likes + comments                          AS total_engagement,
        ROUND((likes + comments) * 1.0 / NULLIF(views, 0), 4) AS engagement_rate
    FROM youtube_trending
),
ranked AS (
    SELECT *,
        AVG(engagement_rate) OVER (
            PARTITION BY category
            ORDER BY trending_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_engagement
    FROM base
)
SELECT category, ROUND(AVG(rolling_7d_engagement), 4) AS avg_rolling_engagement
FROM ranked
GROUP BY category
ORDER BY avg_rolling_engagement DESC;
 
-- Feature Engineering: days-to-trend with percentile bucketing
SELECT
    video_id,
    title,
    DATEDIFF(trending_date, DATE(publish_time))        AS days_to_trend,
    NTILE(4) OVER (ORDER BY 
        DATEDIFF(trending_date, DATE(publish_time))
    )                                                   AS trend_speed_quartile
FROM youtube_trending
WHERE DATEDIFF(trending_date, DATE(publish_time)) >= 0;
```
 

 
<div align="center">
 
 
 
</div>
