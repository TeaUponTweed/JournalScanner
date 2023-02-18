# Overview
Simple CLI to create scan-like images from pictures of documents. There are currently several issues with the algorithm but it works _sometimes_.
The eventual goal is to implement this in golang and compile to WASM and run locally in the browser and allow for manual quadrilateral inputs when scanning non-isolated images such as a recipe book.

We take in a image like this:
![example input](/test_images/example_input.jpg)
And ideally output an image like this:
![example output](/test_images/example_output.jpg)

# Usage
Install by running `make install`
Then run
`scan path/to/image.jpeg path/to/scan.jpeg`

# TODO
- Use color (white-ish -> other) to improve edge detection
- Track original corner orientation so that document is not flipped
- More rigorous down-sampling for parts of the algorithm
- Implement min-cut to remove areas of darkness
- Can this be a pure DL algorithm? Perhaps using "SmartDoc-QA: A dataset for quality assessment of smartphone captured document images - single and multiple distortions"
- Typehint / pylint / mypy
