import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

#############################################################################################################################
# BƯỚC 0: UPLOAD DATA
#############################################################################################################################

# Bước 1: Upload dữ liệu
def upload_data(dir_file = "data_analysis/OnlineRetail.csv", encoding='latin1', date_='InvoiceDate', price = 'UnitPrice'):
    # 2. Upload data => name file: House_data.xlsx
    
    df = pd.read_csv(dir_file, encoding=encoding)

    # convert object to datetime
    df[date_] = pd.to_datetime(df[date_])

    # convert object to float
    df[price] = df[price].apply(lambda col:pd.to_numeric(col, errors='coerce'))

    return df



#############################################################################################################################
# BƯỚC 1: THỰC HIỆN THÊM THỜI GIAN CHO BÀI TOÁN
#############################################################################################################################


# Bước 2: Get tháng
def get_month(x): 
    return dt.datetime(x.year, x.month, 1)


def add_time(df, date_='InvoiceDate', id_customer='CustomerID', name_output = 'CohortMonth', min_ = 'min'):
    df[date_] = df[date_].apply(get_month)
    grouping = df.groupby(id_customer)[date_]
    df[name_output] = grouping.transform(min_)

    return df

# Thêm hàn ngày tháng năm
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


def get_date(df, date_='InvoiceDate', name_input = 'CohortMonth', name_output ='CohortIndex'):
    invoice_year, invoice_month, _ = get_date_int(df, date_)
    cohort_year, cohort_month, _ = get_date_int(df, name_input)
    years_diff = invoice_year - cohort_year
    months_diff = invoice_month - cohort_month
    df[name_output] = years_diff * 12 + months_diff + 1

    return df

# Tổng hợp bài toán
def get_date_cohort_analysis(df, date_='InvoiceDate', id_customer='CustomerID', name_output = 'CohortMonth', min_ = 'min' , name_output2 ='CohortIndex'):
    df_new = add_time(df, date_=date_, id_customer=id_customer, name_output = name_output, min_ = min_)
    results = get_date(df_new, date_=date_, name_input = name_output, name_output =name_output2)
    return results


#############################################################################################################################
# BƯỚC 2: BẢNG ĐẾM SỐ LẦN KHÁCH HÀNG MUA VÀ QUAY LẠI MUA SAU 1 THÁNG
#############################################################################################################################

def cohort_counts(df, CohortMonth='CohortMonth', CohortIndex = 'CohortIndex', id_customer='CustomerID', variable1 = 'Quantity', variable2='UnitPrice', name_variable_new = 'revenue'):
    # Count monthly active customers from each cohort
    df[name_variable_new] = df[variable1] * df[variable2]
    grouping = df.groupby([CohortMonth, CohortIndex])
    cohort_data = grouping[id_customer].apply(pd.Series.nunique)
    cohort_data = cohort_data.reset_index()
    cohort_counts = cohort_data.pivot(index=CohortMonth, columns=CohortIndex, values=id_customer)


    cohort_sizes = cohort_counts.iloc[:,0]
    retention_visu = cohort_counts.divide(cohort_sizes, axis=0)
    retention = retention_visu.round(3) * 100
    

    grouping = df.groupby(['CohortMonth', 'CohortIndex'])

    cohort_data = grouping[variable1].mean()
    cohort_data = cohort_data.reset_index()
    average_quantity = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values=variable1)
    average_quantity = average_quantity.round(1)

    cohort_data = grouping[variable2].mean()
    average_revenue = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values=variable2)
    average_revenue.round(1)

    return cohort_counts, retention, retention_visu, average_quantity, average_revenue




#############################################################################################################################
# BƯỚC 3: TRỰC QUAN HÓA DỮ LIỆU
#############################################################################################################################
def plot_visualation(df, title = 'Retention rates', width_ = 22, height_ = 24):
    result = (
        plt.figure(figsize=(width_, height_)),
        plt.title(title),
        sns.heatmap(data = df, annot = True, fmt = '.0%', vmin = 0.0, vmax = 0.5, cmap = 'BuGn'),
        plt.show()
    )
    return result

def plot_visualation2(df, title = 'Retention rates', width_ = 22, height_ = 24):
    result = (
        plt.figure(figsize=(width_, height_)),
        plt.title(title),
        sns.heatmap(data = df, annot = True, cmap = 'BuGn'),
        plt.show()
    )
    return result

##################################################################################################################################
#
##################################################################################################################################

import numpy as np
import pandas as pd
import datetime as dt

# 2. Upload data => name file: House_data.xlsx
dir_file = "data_analysis/OnlineRetail.csv"
def upload_data_2(dir_file):
    df = pd.read_csv(dir_file, encoding='latin1')
    return df


def cohort_analysis(df):
    # convert object to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # convert object to float
    #df['UnitPrice'] = df['UnitPrice'].apply(lambda x: x.replace(',', '.'))
    df['UnitPrice'] = df['UnitPrice'].apply(lambda col:pd.to_numeric(col, errors='coerce'))

    # Get tháng
    def get_month(x): 
        return dt.datetime(x.year, x.month, 1)

    df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)
    grouping = df.groupby('CustomerID')['InvoiceMonth']
    df['CohortMonth'] = grouping.transform('min')

    # Thêm hàn ngày tháng năm
    def get_date_int(df, column):
        year = df[column].dt.year
        month = df[column].dt.month
        day = df[column].dt.day
        return year, month, day

    invoice_year, invoice_month, _ = get_date_int(df, 'InvoiceMonth')
    cohort_year, cohort_month, _ = get_date_int(df, 'CohortMonth')
    years_diff = invoice_year - cohort_year
    months_diff = invoice_month - cohort_month
    df['CohortIndex'] = years_diff * 12 + months_diff + 1

    df['revenue'] = df['Quantity'] * df['UnitPrice']

    # Count monthly active customers from each cohort
    grouping = df.groupby(['CohortMonth', 'CohortIndex'])
    #cohort_data = grouping.groupby(['CustomerID']).agg('count')
    cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)
    cohort_data = cohort_data.reset_index()
    cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')

    cohort_sizes = cohort_counts.iloc[:,0]
    retention = cohort_counts.divide(cohort_sizes, axis=0)


    grouping = df.groupby(['CohortMonth', 'CohortIndex'])

    cohort_data = grouping['Quantity'].mean()
    cohort_data = cohort_data.reset_index()

    average_quantity = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='Quantity')
    average_quantity.round(1)

    grouping = df.groupby(['CohortMonth', 'CohortIndex'])

    cohort_data = grouping['revenue'].mean()
    cohort_data = cohort_data.reset_index()

    average_revenue = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='revenue')
    average_revenue = average_revenue.round(1)

    return cohort_counts, retention, average_quantity, average_revenue