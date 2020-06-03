#Nhớ install các thư viện cần dùng và tensorflow version 2.0.0a gì đó quên rồi :D
# tri_tue_nhan_tao_Van_Long
# Nếu dùng file pretrain thì chỉ cần chạy python3 cnn_video.py
# Nếu muốn train lại model
## Đầu tiên cần phải load ảnh vào folder dataset/progress (happy,angry,....)
## Chạy python3 resize_image.py (nếu không chạy file này vẫn được)
## Chạy file python3 mtCNN_crop.py để crop faces
## Chạy file cnn_keras_train.py để train model
## kết thúc chạy file python3 cnn_video.py