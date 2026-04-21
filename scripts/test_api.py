import argparse
import base64
import io
import json

import requests
from PIL import Image
from tensorflow.keras.datasets import mnist


def encode_test_image(index: int) -> tuple[str, int]:
    (_, _), (x_test, y_test) = mnist.load_data()
    image = Image.fromarray(x_test[index])
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded, int(y_test[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba endpoint /predict")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/predict",
        help="URL del endpoint /predict",
    )
    parser.add_argument( 
        "--index",
        type=int,
        default=0,
        help="Indice de imagen en MNIST test set",
    )
    args = parser.parse_args()

    b64_image, expected_digit = encode_test_image(args.index)
    response = requests.post(args.url, json={"image": b64_image}, timeout=30)
    print(f"Status code: {response.status_code}")
    payload = response.json()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Etiqueta real esperada (MNIST test): {expected_digit}")


if __name__ == "__main__":
    main()
