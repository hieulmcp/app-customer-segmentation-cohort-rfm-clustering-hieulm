import pandas as pd
import datetime as dt



def rfm_model(dir_file):
    df = pd.read_csv(dir_file, encoding='latin1')

    # convert object to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # convert object to float
    #df['UnitPrice'] = df['UnitPrice'].apply(lambda x: x.replace(',', '.'))
    df['UnitPrice'] = df['UnitPrice'].apply(lambda col:pd.to_numeric(col, errors='coerce'))

    ### 1.1. Lọc dữ liệu: Khi phân tích bài toán RFM thì cần quan tâm đến thời gian: tròn đủ 12 tháng
    df = df.loc[(df['InvoiceDate'] >= '2010-12-10') & (df['InvoiceDate'] < '2011-12-13')]


    ### 1.2. Xây dựng chuỗi thời gian trong bài toán RFM
    # Thêm 1 ngày để chúng ta có thể phân tích 1 ngày gần đây
    snapshot_date = max(df.InvoiceDate) + dt.timedelta(days=1)

    df['revenue'] = df['UnitPrice'] * df['Quantity']




    ### 1.3. Xây dựng bộ dữ liệu chuẩn cho bài toán
    #- Tính toán ngày đăng nhập gần đây nhất
    #- Số hóa đơn 
    #- Tổng giá trị

    # Trung bình của mỗi khách hàng 
    datamart = df.groupby(['CustomerID']).agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'count',
            'revenue': 'sum'})


    # Rename columns for easier interpretation
    datamart.rename(columns = {'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'revenue': 'MonetaryValue'}, inplace=True)



    ### 1.4. Gán nhãn dữ liệu theo tứ phân vị
    # Gán nhãn cho tần suất đăng nhập, tần xuất giao dịch
    r_labels = range(3, 0, -1); f_labels = range(1, 4)
    # Gán nhãn cho các nhóm phân vị bằng nhau
    r_groups = pd.qcut(datamart['Recency'], q=3, labels=r_labels)
    # Gán nhãn cho các nhóm phân vị bằng nhau
    f_groups = pd.qcut(datamart['Frequency'], q=3, labels=f_labels)
    # Thêm cột gãn nhãn
    datamart = datamart.assign(R=r_groups.values, F=f_groups.values)


    ### 1.5. Tính điểm RFM
    #- Thêm nhãn cho cột MonetaryValue
    #- Gán nhãn cho cột trên theo số
    #- Thêm cột M
    #- Tính toán RFM_Score

    # Thêm nhãn cho cột MonetaryValue
    m_labels = range(1, 4)
    # Thêm nhãn bằng nhau cho cột MonetaryValue
    m_groups = pd.qcut(datamart['MonetaryValue'], q=3, labels=m_labels)
    # Thêm nhãn cho cột M
    datamart = datamart.assign(M=m_groups.values)
    # Tính RFM_Score
    datamart['RFM_Score'] = datamart[['R','F','M']].sum(axis=1)

    ### 1.6. Đánh giá khách hàng theo tiêu chí giả định như sau
    #- Tạo các phân đoạn có tên Trên cùng, Trung bình, Thấp
    #- Nếu điểm RFM lớn hơn hoặc bằng 10, cấp độ phải là "Top"
    #- Nếu nó nằm trong khoảng từ 6 đến 10 thì nó phải là "Trung bình" và nếu không thì nó phải là "Thấp"

    # Xác định phân cấp 
    def rfm_level(df):
        if df['RFM_Score'] >= 10:
            return 'Bạch kim'
        elif (df['RFM_Score'] >= 7) and (df['RFM_Score'] < 10):
            return 'Vàng'
        elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 7):
            return 'Bạc'
        else:
            return 'Đồng'


    # Thêm 1 cột phân cấp
    datamart['RFM_Level'] = datamart.apply(rfm_level, axis=1)
    ### 1.7. Tính toán giá trị trung bình cho mỗi phân cấp - level và trả về mỗi nhóm
    #- Thứ nhất không có làm việc bộ phận khảo sát thị trường
    #- Thứ 2 thang điểm dựa trên kinh nghiệm bản thân để làm

    rfm_level_agg = datamart.groupby('RFM_Level').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': ['mean', 'count']
    }).round(0)
    rfm_level_agg.columns = rfm_level_agg.columns.droplevel()
    rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean','Count']
    rfm_level_agg['Percent'] = round((rfm_level_agg['Count']/rfm_level_agg.Count.sum())*100,2)
    rfm_level_agg['Percent_Recency'] = round((rfm_level_agg['RecencyMean']/rfm_level_agg.Count.sum())*100,2)
    rfm_level_agg['Percent_Frequency'] = round((rfm_level_agg['FrequencyMean']/rfm_level_agg.Count.sum())*100,2)
    rfm_level_agg['Percent_MonetaryValue'] = round((rfm_level_agg['MonetaryMean']/rfm_level_agg.Count.sum())*100,2)
    # Reset the index
    rfm_level_agg = rfm_level_agg.reset_index()
    rfm_level_agg
    datamart1 = datamart.reset_index()
    datamart1
    return datamart1, rfm_level_agg, datamart