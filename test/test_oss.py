import oss2
import os
from oss2.credentials import EnvironmentVariableCredentialsProvider
from itertools import islice


def set_oss_environment_variables():
    os.environ["OSS_ACCESS_KEY_ID"] = "your access key id"
    os.environ["OSS_ACCESS_KEY_SECRET"] = "your access key secret"


def init_oss(bucket_name, endpoint):
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    return bucket


def upload_file(bucket, path):
    object_name = os.path.basename(path)
    bucket.put_object_from_file(object_name, path)
    print(f"文件 {path} 上传为 {object_name} 成功")


def download_file(bucket, path):
    object_name = os.path.basename(path)
    bucket.get_object_to_file(object_name, path)
    print(f"文件 {object_name} 下载到 {path} 成功")


def list_files(bucket, max_files=10):
    print("存储空间中的文件：")
    file_list = []
    for b in islice(oss2.ObjectIterator(bucket), max_files):
        print(b.key)
        file_list.append(b.key)
    return file_list


def delete_file(bucket, path):
    object_name = os.path.basename(path)  # 直接从路径中获取文件名
    bucket.delete_object(object_name)
    print(f"文件 {object_name} 删除成功")


def list_all_files(bucket):
    print("存储空间中的所有文件：")
    file_list = []
    for obj in oss2.ObjectIterator(bucket):
        print(obj.key)
        file_list.append(obj.key)
    return file_list


def main():
    bucket_name = "zrzrzr"
    endpoint = "oss-cn-beijing.aliyuncs.com"
    local_file_path = "pdf_with_image.pdf"

    set_oss_environment_variables()

    bucket = init_oss(bucket_name, endpoint)
    upload_file(bucket, local_file_path)
    list_all_files(bucket)


if __name__ == "__main__":
    main()
