# Sales Pitch Generator from PDF Content

This project is a Streamlit-based application that generates a compelling sales pitch from the content of an uploaded PDF document. It uses the Groq language model for pitch generation and HuggingFace embeddings for document similarity search.

## Features

- Upload a PDF file and extract its content.
- Enter key points to focus on in the sales pitch.
- Generate vector embeddings from the PDF content.
- Generate a sales pitch tailored to the provided key points.
- Display relevant sections of the document used for generating the pitch.

## Requirements

- Python 3.x
- Streamlit
- LangChain
- Groq API key
- HuggingFace Embeddings
- PyPDFLoader
- FAISS
- Brave Browser (optional, replaceable with Chrome)
- ChromeDriver or BraveDriver

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/bhanu-pratap-rana/sales-pitch-generator.git
    cd sales-pitch-generator
    ```

2. Install the required Python packages:
    ```sh
    pip install streamlit langchain langchain_groq langchain_community langchain_huggingface PyPDFLoader faiss-cpu
    ```

3. Download and install Brave Browser or Chrome Browser if you haven't already.

4. Download the ChromeDriver or BraveDriver from [here](https://sites.google.com/a/chromium.org/chromedriver/downloads) and place it in your project directory.

## Usage

1. Open the `app.py` file and replace the placeholder Groq API key with your own:
    ```python
    GROQ_API_KEY = "YOUR_API_KEY"
    ```

2. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

3. In the web interface:
    - Upload a PDF file containing the content you want to generate a sales pitch from.
    - Enter the key points you want to focus on in the sales pitch.
    - Click "Generate Document Embeddings" to process the PDF and create embeddings.
    - If embeddings are successfully created, click "Generate Sales Pitch" to generate and display the pitch.

## Notes

- The script includes caching for the language model and embeddings to improve performance.
- Make sure to comply with LinkedIn's terms of service when using this script.
- You may need to adjust the sleep times based on your internet connection speed for accurate processing.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

## Contact

For any questions or issues, please open an issue on GitHub.

---

Happy pitching!
