import os
import tempfile
from pathlib import Path
from PIL import Image

def test_score_with_gpt4o():
    # Import the function under test
    from t2v_benchmark import score_with_gpt4o  # replace with actual module file

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY before running the test")

    # Create a temporary image
    tmpdir = tempfile.mkdtemp()
    img_path = Path(tmpdir) / "test.png"
    Image.new("RGB", (64, 64), color="blue").save(img_path)

    # Build a minimal dataset
    dataset = [
        {
            "images": [str(img_path)],
            "texts": ["A small blue square image."]
        }
    ]

    # Call the scoring function
    scores = score_with_gpt4o(dataset=dataset, batch_size=1, openai_key=key)

    # Basic assertions
    assert scores is not None, "No scores returned"
    assert len(scores) == len(dataset), "Score length mismatch"
    print("Test passed. Scores:", scores)


if __name__ == "__main__":
    test_score_with_gpt4o()
