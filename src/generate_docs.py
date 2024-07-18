import os

# Define the path to the directory containing the Plotly HTML files
html_directory = "docs"

# Define the template for the index.html file
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LSTM Price Prediction</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }}
        header {{
            background-color: #4CAF50;
            color: white;
            padding: 1em 0;
            text-align: center;
        }}
        main {{
            padding: 1em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1em;
        }}
        th, td {{
            padding: 0.75em;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        ul {{
            list-style-type: none;
            padding: 0;
        }}
        li {{
            background: white;
            margin: 0.5em 0;
            padding: 1em;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        li a {{
            text-decoration: none;
            color: #333;
        }}
        li a:hover {{
            color: #4CAF50;
        }}
        .search-bar {{
            margin-bottom: 1em;
        }}
        .search-bar input {{
            width: 100%;
            padding: 0.5em;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}
    </style>
    <script>
        function searchGraphs() {{
            let input = document.getElementById('searchInput').value.toLowerCase();
            let table = document.getElementById('graphTable');
            let tr = table.getElementsByTagName('tr');
            for (let i = 1; i < tr.length; i++) {{
                let td = tr[i].getElementsByTagName('td')[0];
                if (td) {{
                    let txtValue = td.textContent || td.innerText;
                    if (txtValue.toLowerCase().indexOf(input) > -1) {{
                        tr[i].style.display = "";
                    }} else {{
                        tr[i].style.display = "none";
                    }}
                }}
            }}
        }}
    </script>
</head>
<body>
    <header>
        <h1>Available Plotly Graphs</h1>
    </header>
    <main>
        <div class="search-bar">
            <input type="text" id="searchInput" onkeyup="searchGraphs()" placeholder="Search for graphs...">
        </div>
        <table id="graphTable">
            <tr>
                <th>Graph</th>
                <th>Features</th>
            </tr>
            {rows}
        </table>
    </main>
</body>
</html>
"""


def extract_symbol_from_filename(filename):
    """
    Extract the symbol from the filename.

    Args:
        filename (str): The filename.

    Returns:
        str: The extracted symbol.
    """
    return filename.split("_")[0]


def extract_features_from_filename(filename):
    """
    Extract features from the filename.

    Args:
        filename (str): The filename.

    Returns:
        str: The extracted features.
    """
    parts = filename.split("_")
    if len(parts) > 2:
        return ", ".join(parts[1:-1])
    return "Unknown"


def generate_index_html():
    # Get a list of all HTML files in the directory
    files = [f for f in os.listdir(
        html_directory) if f.endswith("_predictions.html")]

    # Generate the list of links and rows
    rows = []
    for file in files:
        symbol = extract_symbol_from_filename(file)
        features = extract_features_from_filename(file)
        rows.append(
            f'<tr><td><a href="{file}" target="_blank">{symbol}</a></td><td>{features}</td></tr>'
        )

    # Create the final HTML content
    html_content = INDEX_TEMPLATE.format(rows="\n".join(rows))

    # Write the HTML content to index.html
    with open(os.path.join(html_directory, "index.html"), "w") as f:
        f.write(html_content)
    print(f"index.html generated with {len(files)} links.")


if __name__ == "__main__":
    generate_index_html()
