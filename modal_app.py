from modal import Image
from modal import Stub, web_endpoint

image = Image.debian_slim("3.7").pip_install_from_requirements("", extra_index_url="").apt_install("git")
stub = Stub()

@stub.function(image=image)
@web_endpoint()
def run(image_base64):
    from styleyourhair import setup, process_image
    pass