import requests
import os

url = "http://127.0.0.1:5001/convert_pdf"
pdf_file_path = "pdf_with_image.pdf"


def test_pdf_conversion():
    with open(pdf_file_path, "rb") as pdf_file:
        files = {"pdf_file": pdf_file}
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            with open(f"{pdf_file_path.split('.')[0]}.md", "w") as f:
                f.write(result["full_text"])
            if result.get("images") is None:
                return
            else:
                os.makedirs(f"{pdf_file_path.split('.')[0]}_images", exist_ok=True)
                for i, image in enumerate(result["images"]):
                    with open(f"{pdf_file_path.split('.')[0]}/{i}.png", "wb") as f:
                        f.write(image)
            print("Success saved")
        else:
            print("Failed to convert PDF")


if __name__ == "__main__":
    test_pdf_conversion()
