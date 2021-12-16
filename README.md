CUSTOMER SEGMENTATION COHORT RFM CLUSTERING

TỔNG QUAN VỀ HỆ THỐNG DỮ LIỆU 		
		
		
	1. Mục đích	
		- Công ty X mong muốn có thể bán được nhiều sản phẩm hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hai lòng khách hàng
		- Tìm ra giải pháp giúp cải thiện hiệu quả quảng bá, từ đó giúp tăng doanh thu bán hàng, cải thiện mức độ hài lòng của khách hàng
		- Đưa ra chiến dịch quảng cáo, bán hàng, chăm sóc khách hàng phù hợp cho mỗi nhóm khách hàng
		
	2. Vi sao có dự án nào ?	
		 - Ai (Who): Doanh nghiệp là người cần
		- Tại sao (Why): Xây dựng hệ thống phân cụm khách hàng dữa trên các thông tin do công ty cung cấp từ đó có thể giúp công ty xác định các nhóm khách hàng khác nhau để có chiến lược kinh doanh và chăm sóc khách hàng phù hợp
	3. Dữ liệu cung cấp	
		- Toàn bộ dữ liệu được lưu trữ từ tập tin OnlineRetail.csv với 541.909 record chứa tất cả các giao dịch xẩy ra từ ngày 01/12/2010 đến 09/12/2011 đối với bán lẻ trực tuyến
		- Mô tả dữ liệu: https://archive.ics.uci.edu/ml/datasets/online+retail
		
	3. Vấn đề	
		- Công ty X chủ yếu bán các sản phẩm là quà tặng dành cho những dịp đặc biệt.
		- Nhiều khách hàng của công ty là khách hàng bán buôn
		
		
	4. Thách thức và cách tiếp cận - Challenge and Approach	
		- Dữ liệu chỉ lấy dựa trên data của công ty nên không có sự đối chiệu
		- Không tiếp cận với nguồn lực sale và MKT để xem mức độ phân loại khách hàng phu hợp chưa và cần cải thiện gì theo bộ chỉ tiêu phân loại
		
		
	5. Data obtained - Thu thập dữ liệu	
		- Không thông quan nguồn cào data
		- Thông tin được lấy từ file của công ty
		
![image](https://user-images.githubusercontent.com/96172322/146408933-f136b910-728a-40c9-8018-97304cc6df6e.png)
6. Các công việc cần thực hiện	
Bước 1	Thực hiên theo các bước data science process
Bước 2	Áp dụng RFM và các thuật toán phân cụm phù hợp
Bước 3	Đánh giá và report kết quả
7. Đặt ra yêu cầu với bài toán	
	- Phân khúc khách hàng theo country để hiểu hơn về đặc điểm địa lý, nhân khẩu học, tâm lý, hành vi (geographic, demographic, psychographic, behavioral)
	- Doanh thu bán hàng như thế nào ? => Với những nhãn hàng blank thì sẽ chiếm bao nhiêu % doanh thu chưa có mã
	- Nên xem mức độ doanh thu => của từng phân khúc khu vực để đưa và chiến lược muốn tăng khu vực nào nhiều hơn
	
![image](https://user-images.githubusercontent.com/96172322/146408965-eeff1fb6-b63f-4561-bafd-7a6500aa50df.png)



