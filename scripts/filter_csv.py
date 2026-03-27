import pandas as pd 
import os

data_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_csv = os.path.join(data_dir,'data','annotation.csv')

df = pd.read_csv(data_csv)
print(df.iterrows())

def extract_video_id(url):
    if not isinstance(url,str):
        return None
    basename = os.path.basename(url)
    video_id  = os.path.splitext(basename)[0]
    return video_id

def main():
    print(f"Đang lọc dữ liệu từ {data_csv}")
    try:
        df = pd.read_csv(data_csv)
        df_id = df.copy()
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {data_csv}. Hãy kiểm tra lại đường dẫn!")
        return
    
    df_id.dropna(subset=['Input.Video'],inplace=True)
    df_id["video_id"] = df_id['Input.Video'].apply(extract_video_id) 
    df_id = df_id.drop_duplicates(subset=['video_id'], keep='first')
    print(f"Số video duy nhất sau khi lọc: {len(df_id)}")
    df_id.to_csv(os.path.join(data_dir,'data','train_labels.csv'), index=False)
    print(f"Đã lưu file train_labels.csv")

if __name__ == "__main__":
    main()

