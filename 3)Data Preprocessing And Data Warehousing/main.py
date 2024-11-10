import pandas as pd
try:
    product_df = pd.read_csv('Online_retail.csv', encoding='ISO-8859-1')
    store_df = pd.read_csv('train.csv', encoding='ISO-8859-1')
    salesperson_df = pd.read_csv('Salesperson.csv', encoding='ISO-8859-1')
    sales_fact_df = pd.read_csv('Global_Superstore2.csv', encoding='ISO-8859-1')
except UnicodeDecodeError as e:
    print(f"Error reading file: {e}")

# Display the first few rows of each dataset to verify the load
print("Product Dimension:")
print(product_df.head())

print("\nStore Dimension:")
print(store_df.head())

print("\nSalesperson Dimension:")
print(salesperson_df.head())

print("\nSales Fact:")
print(sales_fact_df.head())


product_df = product_df[['StockCode', 'Description', 'UnitPrice']]

store_df = store_df[['Order ID', 'Order Date', 'Product ID', 'Sales', 'Region']]

salesperson_df = salesperson_df[['EmployeeNumber', 'EmployeeCount', 'Department', 'JobRole']]


sales_fact_df = sales_fact_df[['Product Name', 'Sales', 'Quantity', 'Discount', 'Profit']]


product_df.dropna(inplace=True)
store_df.dropna(inplace=True)
salesperson_df.dropna(inplace=True)
sales_fact_df.dropna(inplace=True)



product_df['UnitPrice'] = product_df['UnitPrice'].astype(float)
sales_fact_df['Sales'] = sales_fact_df['Sales'].astype(float)
sales_fact_df['Quantity'] = sales_fact_df['Quantity'].astype(int)
sales_fact_df['Discount'] = sales_fact_df['Discount'].astype(float)
sales_fact_df['Profit'] = sales_fact_df['Profit'].astype(float)

from sqlalchemy import create_engine, Table, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base

# Create a SQLite engine and base class
engine = create_engine('sqlite:///data_warehouse.db')
Base = declarative_base()

# Define dimension tables
class Product(Base):
    __tablename__ = 'Product'
    ProductID = Column(Integer, primary_key=True)
    Description = Column(String)
    UnitPrice = Column(Float)
    StockCode = Column(String)

class Store(Base):
    __tablename__ = 'Store'
    StoreID = Column(Integer, primary_key=True)
    StoreName = Column(String)
    City = Column(String)
    State = Column(String)
    Region = Column(String)

class Salesperson(Base):
    __tablename__ = 'Salesperson'
    EmployeeID = Column(Integer, primary_key=True)
    Name = Column(String)
    Region = Column(String)
    JobRole = Column(String)


# Define the fact table
class SalesFact(Base):
    __tablename__ = 'SalesFact'
    OrderID = Column(Integer, primary_key=True)
    OrderDate = Column(Date)
    ProductID = Column(Integer)
    StoreID = Column(Integer)
    EmployeeID = Column(Integer)
    Sales = Column(Float)
    Quantity = Column(Integer)
    Discount = Column(Float)
    Profit = Column(Float)


# Create tables
Base.metadata.create_all(engine)


from sqlalchemy import inspect

# Inspect the tables
inspector = inspect(engine)

# List tables
print("Tables in the database:")
for table_name in inspector.get_table_names():
    print(table_name)

# Inspect columns in SalesFact table
columns = inspector.get_columns('SalesFact')
print("\nColumns in SalesFact:")
for column in columns:
    print(f"Column: {column['name']}, Type: {column['type']}")


# Example query: Total sales per day
query = """
    SELECT OrderDate, SUM(Sales) as TotalSales
    FROM SalesFact
    GROUP BY OrderDate
    ORDER BY OrderDate
"""
sales_trends = pd.read_sql(query, engine)
print("Sales Trends:")
print(sales_trends)

# Check indexes in the database
indexes = inspector.get_indexes('SalesFact')
print("\nIndexes on SalesFact:")
for index in indexes:
    print(f"Index Name: {index['name']}, Columns: {index['column_names']}")
