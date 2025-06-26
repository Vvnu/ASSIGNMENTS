import pandas as pd

# Read the CSV file
df = pd.read_csv('Smartphones_cleaned_dataset.csv')

# Display basic info about the dataset
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("="*60)

# # Clean the data (remove duplicates)
df_clean = df.drop_duplicates(subset=['brand_name', 'model'])
print(f"After removing duplicates: {df_clean.shape[0]} rows")
print("="*60)

# 1. PRICE-BASED FILTERS
print("1. PRICE-BASED FILTERS:")
print("-" * 30)

# Budget phones (under ₹20,000)
budget_phones = df_clean[df_clean['price'] < 20000]
print(f"\nBudget phones (< ₹20,000): {len(budget_phones)} phones")
if len(budget_phones) > 0:
    print(budget_phones[['brand_name', 'model', 'price', 'rating']].to_string(index=False))

# Mid-range phones (₹20,000 - ₹30,000)
mid_range = df_clean[(df_clean['price'] >= 20000) & (df_clean['price'] <= 30000)]
print(f"\nMid-range phones (₹20,000 - ₹30,000): {len(mid_range)} phones")
if len(mid_range) > 0:
    print(mid_range[['brand_name', 'model', 'price', 'rating']].to_string(index=False))

# Premium phones (> ₹30,000)
premium_phones = df_clean[df_clean['price'] > 30000]
print(f"\nPremium phones (> ₹30,000): {len(premium_phones)} phones")
if len(premium_phones) > 0:
    print(premium_phones[['brand_name', 'model', 'price', 'rating']].to_string(index=False))

print("\n" + "="*60)

# 2. BRAND-BASED FILTERS
print("2. BRAND-BASED FILTERS:")
print("-" * 30)

# Show all available brands first
print(f"Available brands: {df_clean['brand_name'].unique()}")

# Filter by specific brand
brand_to_filter = 'oneplus'  # Change this to any brand you want
brand_phones = df_clean[df_clean['brand_name'] == brand_to_filter]
print(f"\n{brand_to_filter.title()} phones: {len(brand_phones)} phones")
if len(brand_phones) > 0:
    print(brand_phones[['brand_name', 'model', 'price', 'rating']].to_string(index=False))

# Multiple brands filter
selected_brands = ['samsung', 'xiaomi', 'realme']  # Modify as needed
multi_brand = df_clean[df_clean['brand_name'].isin(selected_brands)]
print(f"\nPhones from {selected_brands}: {len(multi_brand)} phones")
if len(multi_brand) > 0:
    print(multi_brand[['brand_name', 'model', 'price', 'rating']].to_string(index=False))

print("\n" + "="*60)

# 3. RATING-BASED FILTERS
print("3. RATING-BASED FILTERS:")
print("-" * 30)

# High-rated phones
min_rating = 85  # Change this value as needed
high_rated = df_clean[df_clean['rating'] >= min_rating]
print(f"\nHigh-rated phones (≥{min_rating}): {len(high_rated)} phones")
if len(high_rated) > 0:
    print(high_rated[['brand_name', 'model', 'rating', 'price']].to_string(index=False))

# Rating range filter
rating_min, rating_max = 80, 85
rating_range = df_clean[(df_clean['rating'] >= rating_min) & (df_clean['rating'] <= rating_max)]
print(f"\nPhones with rating {rating_min}-{rating_max}: {len(rating_range)} phones")
if len(rating_range) > 0:
    print(rating_range[['brand_name', 'model', 'rating', 'price']].to_string(index=False))

print("\n" + "="*60)

# 4. FEATURE-BASED FILTERS
print("4. FEATURE-BASED FILTERS:")
print("-" * 30)

# 5G phones
phones_5g = df_clean[df_clean['has_5g'] == True]  # or == 'TRUE' depending on data format
print(f"\n5G phones: {len(phones_5g)} phones")
if len(phones_5g) > 0:
    print(phones_5g[['brand_name', 'model', 'has_5g', 'price']].head().to_string(index=False))

# NFC enabled phones
nfc_phones = df_clean[df_clean['has_nfc'] == True]  # or == 'TRUE'
print(f"\nNFC enabled phones: {len(nfc_phones)} phones")
if len(nfc_phones) > 0:
    print(nfc_phones[['brand_name', 'model', 'has_nfc', 'price']].to_string(index=False))

# Operating system filter
android_phones = df_clean[df_clean['os'] == 'android']
print(f"\nAndroid phones: {len(android_phones)} phones")

ios_phones = df_clean[df_clean['os'] == 'ios']
print(f"iOS phones: {len(ios_phones)} phones")
if len(ios_phones) > 0:
    print(ios_phones[['brand_name', 'model', 'os', 'price']].to_string(index=False))

print("\n" + "="*60)

# 5. SPECIFICATIONS-BASED FILTERS
print("5. SPECIFICATIONS-BASED FILTERS:")
print("-" * 30)

# RAM filter
min_ram = 8  # GB
high_ram = df_clean[df_clean['ram_capacity'] >= min_ram]
print(f"\nPhones with {min_ram}GB+ RAM: {len(high_ram)} phones")
if len(high_ram) > 0:
    print(high_ram[['brand_name', 'model', 'ram_capacity', 'price']].to_string(index=False))

# Storage filter
min_storage = 128  # GB
high_storage = df_clean[df_clean['internal_memory'] >= min_storage]
print(f"\nPhones with {min_storage}GB+ storage: {len(high_storage)} phones")
if len(high_storage) > 0:
    print(high_storage[['brand_name', 'model', 'internal_memory', 'price']].head().to_string(index=False))

# Screen size filter
min_screen = 6.5  # inches
large_screen = df_clean[df_clean['screen_size'] >= min_screen]
print(f"\nPhones with {min_screen}\"+ screen: {len(large_screen)} phones")
if len(large_screen) > 0:
    print(large_screen[['brand_name', 'model', 'screen_size', 'price']].to_string(index=False))

# High refresh rate
high_refresh = df_clean[df_clean['refresh_rate'] >= 120]
print(f"\nHigh refresh rate phones (120Hz+): {len(high_refresh)} phones")
if len(high_refresh) > 0:
    print(high_refresh[['brand_name', 'model', 'refresh_rate', 'price']].head().to_string(index=False))

print("\n" + "="*60)

# 6. COMPLEX/COMBINED FILTERS
print("6. COMPLEX/COMBINED FILTERS:")
print("-" * 30)

# Best value phones: Good rating + reasonable price
best_value = df_clean[(df_clean['rating'] >= 82) & (df_clean['price'] < 25000)]
print(f"\nBest value phones (Rating ≥82 & Price <₹25,000): {len(best_value)} phones")
if len(best_value) > 0:
    print(best_value[['brand_name', 'model', 'rating', 'price']].to_string(index=False))

# Gaming phones: High RAM + High refresh rate + Good processor
gaming_phones = df_clean[
    (df_clean['ram_capacity'] >= 8) & 
    (df_clean['refresh_rate'] >= 120) & 
    (df_clean['processor_brand'].isin(['snapdragon', 'dimensity', 'bionic']))
]
print(f"\nGaming phones (8GB+ RAM, 120Hz+, Good processor): {len(gaming_phones)} phones")
if len(gaming_phones) > 0:
    print(gaming_phones[['brand_name', 'model', 'ram_capacity', 'refresh_rate', 'processor_brand', 'price']].to_string(index=False))

# Photography phones: High megapixel camera
photography_phones = df_clean[df_clean['primary_camera_rear'] >= 100]
print(f"\nPhotography phones (100MP+ rear camera): {len(photography_phones)} phones")
if len(photography_phones) > 0:
    print(photography_phones[['brand_name', 'model', 'primary_camera_rear', 'price']].to_string(index=False))

# Budget 5G phones
budget_5g = df_clean[(df_clean['price'] < 20000) & (df_clean['has_5g'] == True)]
print(f"\nBudget 5G phones (<₹20,000 with 5G): {len(budget_5g)} phones")
if len(budget_5g) > 0:
    print(budget_5g[['brand_name', 'model', 'price', 'has_5g']].to_string(index=False))

print("\n" + "="*60)

# 7. SAVE FILTERED DATA
print("7. SAVING FILTERED DATA:")
print("-" * 30)

# Save different filtered datasets to separate CSV files
budget_phones.to_csv('budget_phones.csv', index=False)
print(f"Budget phones saved to 'budget_phones.csv' ({len(budget_phones)} records)")

high_rated.to_csv('high_rated_phones.csv', index=False)
print(f"High-rated phones saved to 'high_rated_phones.csv' ({len(high_rated)} records)")

best_value.to_csv('best_value_phones.csv', index=False)
print(f"Best value phones saved to 'best_value_phones.csv' ({len(best_value)} records)")

print("\n" + "="*60)

# 8. CUSTOM FILTER FUNCTION
print("8. CUSTOM FILTER FUNCTION:")
print("-" * 30)

def filter_phones(df, max_price=None, min_rating=None, brands=None, min_ram=None, os=None):
    """
    Custom function to filter phones based on multiple criteria
    
    Parameters:
    - max_price: Maximum price limit
    - min_rating: Minimum rating required
    - brands: List of preferred brands
    - min_ram: Minimum RAM required (GB)
    - os: Operating system preference ('android' or 'ios')
    """
    filtered_df = df.copy()
    
    if max_price:
        filtered_df = filtered_df[filtered_df['price'] <= max_price]
    
    if min_rating:
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    if brands:
        filtered_df = filtered_df[filtered_df['brand_name'].isin(brands)]
    
    if min_ram:
        filtered_df = filtered_df[filtered_df['ram_capacity'] >= min_ram]
    
    if os:
        filtered_df = filtered_df[filtered_df['os'] == os]
    
    return filtered_df

# Example usage of custom filter
custom_filtered = filter_phones(
    df_clean, 
    max_price=30000, 
    min_rating=82, 
    brands=['oneplus', 'xiaomi', 'realme'],
    min_ram=6
)

print(f"\nCustom filtered phones: {len(custom_filtered)} phones")
# if len(custom_filtered) > 0:
#     print(custom_filtered[['brand_name', 'model', 'price', 'rating', 'ram_capacity']].to_string(index=False))

# print(f"\nFiltering complete! Check the generated CSV files for specific filtered datasets.")


















# # print(df_clean)

# # # Display the dataframe
# # print(df.shape)  # Shows first 5 rows
# # print(df.head())  # In Jupyter notebook
# # print(df.info())           # Column info and data types
# # print(df.describe())       # Statistical summary
# # print(df.shape)           # Number of rows and columns
# # print(df.columns)         # Column names
# # print(df.dtypes)          # Data types of each column