###########################################################################################
# 1. Bài toán Recency - frequency Monetary value analysis
###########################################################################################
import streamlit as st
import pandas as pd
import lib.step17_RFM as rfm
import lib.step16_cohort_analysis as coh
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import squarify
import plotly.express as px
import seaborn as sns
# Import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy import stats
from sklearn.metrics import silhouette_score
plt.style.use('seaborn')
st.set_option('deprecation.showPyplotGlobalUse', False)




menu = ["Summary","Cohort Analysis","Recency-Frequency-Monetary Value analysis", "Clustering analysis", "Kết luật và hướng phát triển hệ thống đề xuất" ]
choice = st.sidebar.selectbox('Menu',menu)
if choice == "Summary":
    st.markdown("<h1 style='text-align: center; color: Coral;'>CUSTOMER SEGMENTATION - ONLINE RETAIL</h1>", unsafe_allow_html=True)
    st.image('picture/summary.png')
    
    st.markdown("<h2 style='text-align: left; color: Yellow;'>A. NHÌN CHUNG VỀ CUSTOMER - ONLINE RETAIL</h2>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: left; color: Yellow;'>1. Tổng quan về customer segmentation - online retail</h5>", unsafe_allow_html=True)
    st.markdown("- Những công ty hiện nay điều quan trọng nhất vẫn là hiễu rõ khách hàng của họ đến mức có thể đoán được trước nhu cầu của khách hàng")
    st.markdown("- Các nhà phân tích dữ liệu đóng một vai trò quan trọng trong việc mở khóa những thông tin chi tiết chuyên sâu này và phân khúc khách hàng để phục vụ họ tốt hơn")
    st.markdown("-  Kỹ thuật thực tế về phân khúc khách hàng và phân tích hành vi, sử dụng tập dữ liệu thực có chứa các giao dịch của khách hàng từ một nhà bán lẻ trực tuyến")
    st.write(" ")

    st.markdown("<h5 style='text-align: left; color: Yellow;'>2. Những điều sẽ làm trong customer segmentation - online retail</h5>", unsafe_allow_html=True)
    st.markdown("- Đầu tiên bạn sẽ xác định những sản phẩm nào thường xuyên được mua cùng nhau")
    st.markdown("- Chạy phân tích thuần tập để hiểu xu hướng của khách hàng")
    st.markdown("- Cách xây dựng các phân khúc khách hàng dễ hiểu")
    st.markdown("- Sẽ làm cho các phân đoạn của mình mạnh mẽ hơn với phân cụm k-mean, chỉ trong vài dòng mã")
    st.markdown("- Sẽ có thể áp dụng các kỹ thuật phân tích và phân khúc hành vi khách hàng thực tế")
    st.write(" ")

    st.markdown("<h5 style='text-align: left; color: Yellow;'>3. Marketing</h5>", unsafe_allow_html=True)
    st.markdown("- Trong thống kê, tiếp thị và nhân khẩu học, một nhóm các đối tượng có chung một đặc điểm xác định (thường là các đối tượng đã trải qua một sự kiện chung trong một khoảng thời gian đã chọn)")
    st.markdown("- Đôi khi, dữ liệu thuần tập có thể có lợi hơn cho các nhà nhân khẩu học so với dữ liệu thời kỳ")
    st.markdown("- Vì dữ liệu thuần tập được chỉnh sửa theo một khoảng thời gian cụ thể, nên nó thường chính xác hơn")
    st.markdown("- Nó chính xác hơn vì nó có thể được điều chỉnh để truy xuất dữ liệu tùy chỉnh cho một nghiên cứu cụ thể")
    st.write(" ")


    
elif choice == "Cohort Analysis":

    st.markdown("<h1 style='text-align: center; color: Coral;'>Cohort Analysis</h1>", unsafe_allow_html=True)
    st.image('picture/Cohort_Analysis.png')

    st.markdown("<h2 style='text-align: left; color: Yellow;'>1. Cohort analysis là gì ?</h2>", unsafe_allow_html=True)
    st.markdown("- Kỹ thuật phân tích trong MKT tập trung vào việc phân tích hành vi của 1 nhóm người dùng/ khách hàng có chung 1 đặc điểm trong một khoảng thời gian nhất định, từ khám phá ra những hiểu biết sâu sắc về trải nghiệm của những khách hàng để cải thiện những trải nghiệm đó")
    st.markdown("- Lý do khiến cohort analysis trở nên quan trọng là vì nó giúp marketer vướt qua khỏi  hạn chế của các chỉ số trung bình")
    st.markdown("- Phân tích tổ hợp là một công cụ để đo lường mức độ tương tác người dùng theo thời gian. Nó giúp biết liệu mực độ tương tác của người dùng đang thực sự tốt hơn theo thời gian hay chỉ có vẻ cải thiện do tăng trưởng")
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>2. Cách tính Cohort Analysis</h2>", unsafe_allow_html=True)
    st.image('picture/CRR.png')
    st.markdown("- Trong tất cả các ngành này, phân tích cohort thường được sử dụng để xác định lý do tại sao khách hàng rời đi và những gì có thể làm để ngăn họ rời đi. Điều đó đưa chúng ta đến việc tính toán Customer Retention Rate – Tỷ lệ giữ chân khách hàng (Viết tắt là CRR)")
    st.markdown("- Tỷ lệ giữ chân khách hàng được tính bởi công thức này: CRR = ((E-N) / S) X 100")
    st.markdown(" - E – Số lượng khách hàng cuối sử dụng vào cuối kỳ của khoảng giai đoạn.")
    st.markdown(" - N – Số lượng khách hàng có được trong khoảng thời gian đó.")
    st.markdown(" - S – Số lượng khách hàng đầu kỳ (hoặc đầu kỳ).")
    st.markdown("- CRR càng cao có nghĩa là sự trung thành của khách hàng càng lớn. Bằng cách so sánh điểm chuẩn CRR của doanh nghiệp với mức trung bình trong ngành, bạn có thể thấy vị trí của mình về tỷ lệ giữ chân khách hàng. Nếu CRR cho thấy một bức tranh không mấy tốt, biện pháp khắc phục được thực hiện với sự trợ giúp của phân tích dữ liệu – đây là cách phân tích theo nhóm có thể giúp ích.")


    st.write(" ")







    #BƯỚC 0: UPLOAD DATA
    dir_file = "data_analysis/OnlineRetail.csv"
    encoding='latin1'
    date_='InvoiceDate'
    price = 'UnitPrice'
    df = coh.upload_data(dir_file=dir_file, encoding=encoding, date_=date_, price=price)
    #BƯỚC 1: THỰC HIỆN THÊM THỜI GIAN CHO BÀI TOÁN
    cohort_counts, retention, average_quantity, average_revenue = coh.cohort_analysis(df)

    st.markdown("<h2 style='text-align: left; color: Yellow;'>3. Bảng cohort counts</h2>", unsafe_allow_html=True)
    cohort_counts
    

    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>4. Bảng retention</h2>", unsafe_allow_html=True)
    retention
    

    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>5. Bảng average quantity</h2>", unsafe_allow_html=True)
    average_quantity
    

    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>6. Bảng average revenue</h2>", unsafe_allow_html=True)
    average_revenue
   

    st.write(" ")
    st.write(" ")
    st.write(" ")


    st.markdown("<h2 style='text-align: left; color: Yellow;'>7. cohort_counts rates</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(22, 24))
    plt.title('cohort_counts rates')
    sns.heatmap(data = cohort_counts, annot = True, fmt = '.0%', vmin = 0.0, vmax = 0.5, cmap = 'BuGn')
    plt.show()
    st.pyplot()
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>7. Retention rates</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(22, 24))
    plt.title('Retention rates')
    sns.heatmap(data = retention, annot = True, fmt = '.0%', vmin = 0.0, vmax = 0.5, cmap = 'BuGn')
    plt.show()
    st.pyplot()
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>7. Average_quantity rates</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(22, 24))
    plt.title('Average_quantity rates')
    sns.heatmap(data = average_quantity, annot = True, fmt = '.0%', vmin = 0.0, vmax = 0.5, cmap = 'BuGn')
    plt.show()
    st.pyplot()
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>7. Average_revenue rates</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(22, 24))
    plt.title('Average_revenue rates')
    sns.heatmap(data = average_revenue, annot = True, fmt = '.0%', vmin = 0.0, vmax = 0.5, cmap = 'BuGn')
    plt.show()
    st.pyplot()
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>9. Nhận xét và kết luận ?</h2>", unsafe_allow_html=True)
    st.markdown("- Khách hàng sẽ quay lại sau tháng thứ 12 cho ngày đâu tiên từ 33% lên 45% đến ngày thứ 12 thì giảm xuống và doanh thu cũng dữ khoảng 50%")
    st.markdown("- Lí do khiến cohort analysis trở lên quan trọng là vì nó giúp marketer vượt ra khỏi hạn chế của các chỉ số trung bình, giúp marketer có insight rõ ràng hơn và từ đó đưa ra các quyết định chính xác hơn")
    st.markdown("- Sau khi nhìn vào thực tế các bảng phân tích thì ta cần phân loại khách hàng theo cách tính RFM cho phù hợp bài toán")
    st.markdown("- Đề xuất: Nên họp với bộ MKT để xây dựng bộ chỉ tiêu khách hàng dựa trên các tiêu chí của RFM")
    st.write(" ")

elif choice == "Recency-Frequency-Monetary Value analysis":
    st.markdown("<h1 style='text-align: center; color: Coral;'>Recency, Frequency, Monetary Value analysis</h1>", unsafe_allow_html=True)
    st.image('picture/RFM.jpg')
    
    st.markdown("<h2 style='text-align: left; color: Yellow;'>1. RFM analysis</h2>", unsafe_allow_html=True)
    st.markdown("- R: When is the latest purchase date? - Ngày mua gần nhất là ngày nào ?")
    st.markdown("- F: How frequently do they make purchases? - Họ mua hàng với tân suất như thế nào ?")
    st.markdown("- M: How large their average monetary value is made? - Giá trị tiền tệ trung bình của họ lớn như thế nào?)")
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>2. Giả định</h2>", unsafe_allow_html=True)
    st.image('picture/gia_dinh.PNG')
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>3. Mục tiêu và giải pháp RFM</h2>", unsafe_allow_html=True)
    st.image('picture/muc_tieu.PNG')
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>4. Kế hoạch đề ra</h2>", unsafe_allow_html=True)
    st.image('picture/plan_mkt.PNG')
    st.write(" ")


    # 2. Upload data => name file: House_data.xlsx
    dir_file = "data_analysis/OnlineRetail.csv"
    datamart, rfm_level_agg,_ = rfm.rfm_model(dir_file)

    datamart.CustomerID = datamart.CustomerID.astype('int64')
    datamart.RFM_Score = datamart.RFM_Score.astype('int64')
    datamart.MonetaryValue = datamart.MonetaryValue.astype('float')

    segments = datamart['RFM_Level'].unique()
    customerid = datamart['CustomerID'].unique()

    

    #build app filters
    column = st.sidebar.multiselect('Select Segments', segments)
    #column2 = st.sidebar.multiselect('customerID', customerid)
    #manage the multiple field filter
    if column == []:
        data = datamart
    else:
        data = datamart[datamart['customerID'].isin(column)]
    
    #if column2 == []:
    #    data = datamart
    #else:
     #   data = datamart[datamart['customerID'].isin(column2)]


    #st.form(key='Tìm kiếm')
    #st.write("###### Choose customerID")
    #name_item = st.multiselect(data['CustomerID'])
    #submit_button = st.form_submit_button(label='Tìm kiếm')

    st.markdown("<h2 style='text-align: left; color: Yellow;'>5. Bảng RFM theo customerID</h2>", unsafe_allow_html=True)
    data

    st.write(" ")
    st.write(" ")
    st.write(" ")


    st.markdown("<h2 style='text-align: left; color: Yellow;'>6. Bảng RFM theo RFM_Level</h2>", unsafe_allow_html=True)
    test1 = rfm_level_agg.astype(str)
    st.dataframe(test1)

    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>6. Báo cáo RFM</h2>", unsafe_allow_html=True)

    col1,col2, col3 = st.columns(3)
            
    columns = [col1,col2,col3]
            
    for i in range(len(columns)):
        with columns[i]:
            if 3 == 1:
                st.markdown("<h3 style='text-align: left; color: Aqua;'>R score and average R value</h3>", unsafe_allow_html=True)
                plt.bar(rfm_level_agg['RFM_Level'], rfm_level_agg['RecencyMean'], color ='green',width = 0.8)
                plt.xlabel("RFM_Level")
                plt.ylabel("RecencyMean")
                plt.title("R score and average R value")
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

            elif 3==2:
                st.markdown("<h3 style='text-align: left; color: Aqua;'>F score and average F value</h3>", unsafe_allow_html=True)
                # creating the bar plot
                plt.bar(rfm_level_agg['RFM_Level'], rfm_level_agg['FrequencyMean'], color ='green',
                        width = 0.4)
                
                plt.xlabel("RFM_Level")
                plt.ylabel("FrequencyMean")
                plt.title("F score and average F value ")
                plt.show()

                st.pyplot()
            elif 3==3:
                st.markdown("<h6 style='text-align: center; color: Aqua;'>M score and average M value</h6>", unsafe_allow_html=True)
                # creating the bar plot
                plt.bar(rfm_level_agg['RFM_Level'], rfm_level_agg['MonetaryMean'], color ='green',
                        width = 0.4)
                
                plt.xlabel("RFM_Level")
                plt.ylabel("MonetaryMean")
                plt.title("M score and average M value ")
                plt.show()

                st.pyplot()

    st.write(" ")

    # Create our plot and resize it
    st.markdown("<h6 style='text-align: center; color: Aqua;'>Number of the customers in each segment </h6>", unsafe_allow_html=True)
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14,10)

    color_dict = {'GOLD':'yellow','SILVER':'royalblue','BRONZE':'cyan',\
    'LOST':'red','NEW':'green','VIP':'gold'}

    squarify.plot(
        sizes=rfm_level_agg['Count'],
        text_kwargs={'fontsize':12,'weight':'bold','fontname':'sans serif'},
        color=color_dict.values(),
        label=['{}\n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_level_agg.iloc[i]) for i in range(0,len(rfm_level_agg))],
        alpha =0.5
    )
    plt.title("Customer Segments",fontsize =26,fontweight ='bold')
    plt.axis('off')

    plt.savefig('RFM Segment.png')
    plt.show()

    st.pyplot()
    st.write(" ")

    st.markdown("<h6 style='text-align: center; color: Aqua;'>F and M Score of each segment </h6>", unsafe_allow_html=True)
    fig = px.scatter(rfm_level_agg,x = 'FrequencyMean',y='MonetaryMean',size = 'FrequencyMean',color='RFM_Level',hover_name='RFM_Level',size_max=100)
    st.plotly_chart(fig)
    st.write(" ")

    st.markdown("<h6 style='text-align: center; color: Aqua;'>R and M Score of each segment </h6>", unsafe_allow_html=True)
    fig = px.scatter(rfm_level_agg,x = 'RecencyMean',y='MonetaryMean',size = 'FrequencyMean',color='RFM_Level',hover_name='RFM_Level',size_max=100)
    st.plotly_chart(fig)
    st.write(" ")           
            
    st.markdown("<h6 style='text-align: center; color: Aqua;'>R and F Score of each segment </h6>", unsafe_allow_html=True)
    fig = px.scatter(rfm_level_agg,x = 'RecencyMean',y='FrequencyMean',size = 'FrequencyMean',color='RFM_Level',hover_name='RFM_Level',size_max=100)
    st.plotly_chart(fig)
    st.write(" ") 

    st.markdown("<h6 style='text-align: center; color: Aqua;'>3D Score of each segment </h6>", unsafe_allow_html=True)
    fig = px.scatter_3d(datamart,x='Recency',y='Frequency',z='MonetaryValue',color='RFM_Level',opacity=0.5,color_discrete_map=color_dict)
    fig.update_traces(marker = dict(size = 5),selector=dict(mode = 'markers'))

    st.plotly_chart(fig)

    st.markdown("<h2 style='text-align: left; color: Yellow;'>KẾT LUẬN - Ý KIẾN</h2>", unsafe_allow_html=True)
    st.image('picture/KL_RFM.PNG')
    st.write(" ")

elif choice == "Clustering analysis":





    st.markdown("<h1 style='text-align: center; color: Coral;'>Clustering analysis</h1>", unsafe_allow_html=True)
    st.image('picture/kmean.png')

    st.markdown("<h2 style='text-align: left; color: Yellow;'>1. Điều khách hàng muốn và chúng ta muốn ?</h2>", unsafe_allow_html=True)
    st.image('picture/customer.PNG')
    st.write(" ")
   
    st.markdown("<h2 style='text-align: left; color: Yellow;'>2. Chúng ta nên làm gì ?</h2>", unsafe_allow_html=True)
    st.image('picture/database.PNG')
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>3. Thuật toán clustering - Kmean ?</h2>", unsafe_allow_html=True)
    st.image('picture/kmean.jpg')
    st.markdown("<h4 style='text-align: left; color: red;'>Điểm cần chú ý trong Thuật toán clustering - Kmean ?</h4>", unsafe_allow_html=True)
    st.markdown("- K-means giả sử phân phối đồng bộ của các biến và các biến có giá trị trung bình và std bằng nhau")
    st.markdown("- Nếu phân phối không đối xứng")
    st.markdown("- - Log transformation (if all values are positive)")
    st.markdown("- - Thêm giá trị tuyệt đối của giá trị âm thấp nhất vào mỗi quan sát và sau đó với 1 hằng số nhỏ để buộc tất cả các biến là dương")
    st.markdown("- - Use a cube root transformation")
    st.markdown("- - Nếu giá trị trung bình và các biến không bằng nhau, các biến có thể được chuẩn hóa")

    st.write(" ")


    st.markdown("<h2 style='text-align: left; color: Yellow;'>4. K-MEAN</h2>", unsafe_allow_html=True)

    dir_file = "data_analysis/OnlineRetail.csv"
    # 2. Upload data => name file: House_data.xlsx
    #dir_file = "data_analysis/online_retail_data_analysis.csv"
    datamart, rfm_level_agg,_ = rfm.rfm_model(dir_file)
    datamart = datamart[['Recency','Frequency', 'MonetaryValue']]

    #datamart.CustomerID = datamart.CustomerID.astype(str)
    #datamart.RFM_Score = datamart.RFM_Score.astype('int64')
    datamart.MonetaryValue = datamart.MonetaryValue.astype('float')

    def check_skew(df, column):
        skew = stats.skew(df[column])
        skewtest = stats.skewtest(df[column])
        plt.title('Distribution of ' + column)
        sns.distplot(df[column])
        print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
        return 

    st.markdown("<h4 style='text-align: left; color: red;'>Phân phối dữ liệu Recency-Frequency-MonetaryValue</h4>", unsafe_allow_html=True)
    # Plot all 3 graphs together for summary findings
    plt.figure(figsize=(9, 9))

    plt.subplot(3, 1, 1)
    check_skew(datamart,'Recency')
    st.pyplot()

    plt.subplot(3, 1, 2)
    check_skew(datamart,'Frequency')
    st.pyplot()

    plt.subplot(3, 1, 3)
    check_skew(datamart,'MonetaryValue')
    st.pyplot()




    st.markdown("- Nhận xét: dữ liệu lệch phải cần chỉnh sao cho dữ liệu về phân phổi chuẩn")
    st.write(" ")




    st.markdown("<h4 style='text-align: left; color: red;'>Nhìn tổng quan dữ liệu</h4>", unsafe_allow_html=True)
    f = datamart.describe()
    f
    st.markdown("- Nhận xét: Phân phối không đối xứng, nên mean, max, min giữa các biến lệch nhau quá nhiều cần log dữ liêu")
    st.write(" ")




    st.markdown("<h4 style='text-align: left; color: red;'>Chỉnh log dữ liệu với Recency-Frequency-MonetaryValue</h4>", unsafe_allow_html=True)
    # Copy original to new df
    df_rfm_log = datamart.copy()
    # Data Pre-Processing for Negative Value
    df_rfm_log['MonetaryValue'] = (df_rfm_log['MonetaryValue'] - df_rfm_log['MonetaryValue'].min()) + 1
    k = df_rfm_log.describe()
    k
    st.markdown("- Nhận xét: Phân phối không đối xứng, nên mean, max, min giữa các biến lệch nhau quá nhiều cần log dữ liêu")
    st.write(" ")


    st.markdown("<h4 style='text-align: left; color: red;'Scaling data</h4>", unsafe_allow_html=True)
    # Scaling data
    scaler = StandardScaler()
    scaler.fit(df_rfm_log)
    df_rfm_normal = scaler.transform(df_rfm_log)

    df_rfm_normal = pd.DataFrame(df_rfm_normal, index=df_rfm_log.index, columns=df_rfm_log.columns)

    # Check result after standardising
    z = df_rfm_normal.describe().round(3)
    z
    st.markdown("- Nhận xét: Đã điều chỉnh scale data")
    st.write(" ")




    #### 1.3.2. K-means clustering
    st.markdown("<h4 style='text-align: left; color: red;'>Building Customer Personas</h4>", unsafe_allow_html=True)
    st.markdown("- - Elbow criterion (visual method)")
    st.markdown("- - Silhouette Score (math method)")

    def optimal_kmeans(dataset, start=2, end=11):
    
        # Create empty lists to store values for plotting graphs
        n_clu = []
        km_ss = []
        inertia = []

        # Create a for loop to find optimal n_clusters
        for n_clusters in range(start, end):

            # Create cluster labels
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(dataset)

            # Calcualte model performance
            silhouette_avg = round(silhouette_score(dataset, labels, random_state=1), 3)
            inertia_score = round(kmeans.inertia_, 2)

            # Append score to lists
            km_ss.append(silhouette_avg)
            n_clu.append(n_clusters)
            inertia.append(inertia_score)

            print("No. Clusters: {}, Silhouette Score(SS): {}, SS Delta: {}, Inertia: {}, Inertia Delta: {}".format(
                n_clusters, 
                silhouette_avg, 
                round((km_ss[n_clusters - start] - km_ss[n_clusters - start - 1]),3), 
                inertia_score, 
                round((inertia[n_clusters - start] - inertia[n_clusters - start - 1]))),3)

            # Plot graph at the end of loop
            if n_clusters == end - 1:
                plt.figure(figsize=(9,6))

                plt.subplot(2, 1, 1)
                plt.title('Within-Cluster Sum-of-Squares / Inertia')
                sns.pointplot(x=n_clu, y=inertia)

                plt.subplot(2, 1, 2)
                plt.title('Silhouette Score')
                sns.pointplot(x=n_clu, y=km_ss)
                plt.tight_layout()
                st.pyplot()

    optimal_kmeans(df_rfm_log)
    
    
    st.markdown("- Nhận xét: Có thể chọn k = 3,4,5")
    st.write(" ")


    #### 1.3.2. K-means clustering
    st.markdown("<h4 style='text-align: left; color: red;'>Implementing KMeans </h4>", unsafe_allow_html=True)

    def kmeans(normalised_df_rfm, clusters_number, original_df_rfm):
        
        kmeans = KMeans(n_clusters = clusters_number, random_state = 1)
        kmeans.fit(normalised_df_rfm)

        # Extract cluster labels
        cluster_labels = kmeans.labels_
            
        # Create a cluster label column in original dataset
        df_new = original_df_rfm.assign(Cluster = cluster_labels)
        
        # Initialise TSNE
        model = TSNE(random_state=1)
        transformed = model.fit_transform(df_new)
        
        # Plot t-SNE
        plt.title('Flattened Graph of {} Clusters'.format(clusters_number))
        sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster_labels, palette="Set1")
        
        return df_new

    plt.figure(figsize=(9, 9))

    plt.subplot(3, 1, 1)
    df_rfm_k3 = kmeans(df_rfm_normal, 3, datamart)
    st.pyplot()
    plt.subplot(3, 1, 2)
    df_rfm_k4 = kmeans(df_rfm_normal, 4, datamart)
    st.pyplot()
    plt.subplot(3, 1, 3)
    df_rfm_k5 = kmeans(df_rfm_normal, 5, datamart)

    plt.tight_layout()
    plt.savefig('flattened.png', format='png', dpi=1000)
    st.pyplot()
    st.write(" ")

    st.markdown("<h4 style='text-align: left; color: red;'>Build KMeans </h4>", unsafe_allow_html=True)

    def snake_plot(normalised_df_rfm, df_rfm_kmeans, df_rfm_original):
        '''
        Transform dataframe and plot snakeplot
        '''
        # Transform df_normal as df and add cluster column
        normalised_df_rfm = pd.DataFrame(normalised_df_rfm, 
                                        index=datamart.index, 
                                        columns=datamart.columns)
        normalised_df_rfm['Cluster'] = df_rfm_kmeans['Cluster']

        # Melt data into long format
        df_melt = pd.melt(normalised_df_rfm.reset_index(), 
                            id_vars=['Cluster'],
                            value_vars=['Recency', 'Frequency', 'MonetaryValue'], 
                            var_name='Metric', 
                            value_name='Value')

        plt.xlabel('Metric')
        plt.ylabel('Value')
        sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')
        
        return

    plt.figure(figsize=(9, 9))

    plt.subplot(3, 1, 1)
    plt.title('Snake Plot of K-Means = 3')
    snake_plot(df_rfm_normal, df_rfm_k3, datamart)
    st.pyplot()

    plt.subplot(3, 1, 2)
    plt.title('Snake Plot of K-Means = 4')
    snake_plot(df_rfm_normal, df_rfm_k4, datamart)
    st.pyplot()

    plt.subplot(3, 1, 3)
    plt.title('Snake Plot of K-Means = 5')
    snake_plot(df_rfm_normal, df_rfm_k5, datamart)
    st.pyplot()

    st.write(" ")

    st.markdown("<h4 style='text-align: left; color: red;'>KẾT LUẬN - GIẢI PHÁP</h4>", unsafe_allow_html=True)
    st.image('picture/KL.PNG')
    st.write(" ")






        
