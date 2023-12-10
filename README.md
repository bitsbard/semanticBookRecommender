# **Book Recommendations with Text Vectorization and Cosine Similarity**

**I. Data Setup and Exploratory Analysis**
- **Environment Setup**
  - Import essential libraries: pandas, numpy, matplotlib, Google Colab drive.
  - Mount Google Drive for dataset access.
- **Dataset Loading**
  - Define the dataset path from Google Drive.
  - Use pandas to read the dataset, handling problematic lines with `on_bad_lines='skip'`.
  - Exception handling for potential data loading errors.
- **Initial Data Exploration**
  - Display first few rows with `df.head()`.
  - Dataset summary including types and non-null values using `df.info()`.
  - Basic statistical analysis with `df.describe()`.
  - Missing value check using `df.isnull().sum()`.
- **Data Visualization**
  - Histogram of average ratings using matplotlib.
  - Distribution of the number of pages.
  - Bar chart of language distribution.

**II. Feature Engineering, Similarity Computation, and Recommendation Function**
- **Feature Engineering**
  - Convert `publication_date` to datetime format.
  - Extract `publication_year` and `publication_month` as new features.
- **Text Vectorization**
  - Apply TF-IDF vectorization to 'title' and 'authors' columns.
  - Combine these TF-IDF matrices using `hstack`.
- **Data Normalization**
  - Initialize MinMaxScaler.
  - Select and normalize numerical features (`average_rating`, `num_pages`).
  - Convert normalized features to a sparse matrix and combine with text vectorized matrices.
- **Cosine Similarity Computation**
  - Calculate cosine similarity using `cosine_similarity` on the final combined matrix.
  - Create a series mapping book titles to DataFrame indices.
- **Recommendation Function**
  - Define `recommend_books` function using fuzzy matching for title input.
  - Flatten similarity scores, sort, and retrieve top 10 recommendations.
- **Random Book Selection and Recommendation**
  - Select a random book title.
  - Display the random book title.
  - Use the recommendation function to suggest related books.
  - Print recommended books in an organized format.
