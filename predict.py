from cog import BasePredictor, Path, Input
from styleyourhair import setup, process_image


class Predictor(BasePredictor):
    def setup(self):
        setup()

    def predict(self, image: Path = Input(description="Image to buzzcut")) -> Path:
        processed_image = process_image(image.absolute)
        processed_image.save("output.jpg")
        return Path("output.jpg")   