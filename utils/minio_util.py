import yaml
from minio import Minio
from minio.error import S3Error
import os


def load_config(config_path=None):
    """读取 YAML 配置文件（默认使用 common/config_loader 解析出的当前 profile 配置）"""
    if config_path is None:
        from common.config_loader import get_config_path
        config_path = get_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def init_minio_client(config_path=None):
    """初始化 MinIO 客户端"""
    cfg = load_config(config_path)
    minio_conf = cfg["minio"]

    endpoint = minio_conf["endpoint"]
    access_key = minio_conf["access_key"]
    secret_key = minio_conf["secret_key"]

    # secure=False 表示 HTTP，若使用 HTTPS 则改为 True
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    return client


class MinioHelper:
    """MinIO 工具封装"""
    def __init__(self, client: Minio):
        self.client = client

    def ensure_bucket(self, bucket_name):
        """确保 bucket 存在，不存在则创建"""
        found = self.client.bucket_exists(bucket_name)
        if not found:
            self.client.make_bucket(bucket_name)
            print(f"[MinIO] Bucket '{bucket_name}' created.")
        else:
            print(f"[MinIO] Bucket '{bucket_name}' already exists.")

    def upload_file(self, bucket_name, object_name, file_path, content_type="image/jpeg"):
        """上传文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        self.ensure_bucket(bucket_name)
        self.client.fput_object(bucket_name, object_name, file_path, content_type=content_type)
        print(f"[MinIO] Uploaded {file_path} → {bucket_name}/{object_name}")

    def download_file(self, bucket_name, object_name, file_path):
        """下载文件"""
        self.client.fget_object(bucket_name, object_name, file_path)
        print(f"[MinIO] Downloaded {bucket_name}/{object_name} → {file_path}")

    def list_objects(self, bucket_name, prefix=None):
        """列举文件"""
        for obj in self.client.list_objects(bucket_name, prefix=prefix, recursive=True):
            print(f"- {obj.object_name} ({obj.size} bytes)")

    def object_exists(self, bucket_name, object_name):
        """判断文件是否存在"""
        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise



if __name__ == "__main__":
    from common.config_loader import get_config_path

    client = init_minio_client(get_config_path())
    helper = MinioHelper(client)

    bucket = "demo-bucket"
    helper.ensure_bucket(bucket)

    # # 上传示例
    helper.upload_file(bucket, "original/example.yaml", get_config_path())

    # 列出文件
    helper.list_objects(bucket)

    # # 判断是否存在
    # print(helper.object_exists(bucket, "original/example.jpeg"))

    # # 下载示例
    # helper.download_file(bucket, "test/example.txt", "/tmp/example_download.txt")


