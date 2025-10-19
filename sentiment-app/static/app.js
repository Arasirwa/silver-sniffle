// Helper function to determine the appropriate CSS class
const getSentimentClass = (label) => {
    const lowerLabel = label.toLowerCase();
    if (lowerLabel.includes('positive')) return 'positive';
    if (lowerLabel.includes('negative')) return 'negative';
    return 'neutral'; // Catch neutral or other non-binary classes
};

document.getElementById("analyze-btn").onclick = async () => {
    const text = document.getElementById("text-input").value.trim();
    if (!text) return alert("Please enter a tweet!");

    document.getElementById("result").innerHTML = "<p class='placeholder-text'>Analyzing...</p>";

    try {
        const res = await fetch("/predict-text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        if (!res.ok) {
            throw new Error(`Server error: ${res.statusText}`);
        }

        const data = await res.json();
        
        // Apply class-based styling to single analysis results
        const binaryClass = getSentimentClass(data.binary.label);
        const multiClass = getSentimentClass(data.multiclass.label);

        document.getElementById("result").innerHTML = `
            <p><strong>Binary Sentiment:</strong> <span class="${binaryClass}">${data.binary.label}</span> (${(data.binary.confidence * 100).toFixed(1)}%)</p>
            <p><strong>Multiclass Sentiment:</strong> <span class="${multiClass}">${data.multiclass.label}</span> (${(data.multiclass.confidence * 100).toFixed(1)}%)</p>
        `;

    } catch (error) {
        document.getElementById("result").innerHTML = `<p class="negative">Analysis failed: ${error.message}</p>`;
        console.error("Analysis Error:", error);
    }
};

document.getElementById("upload-btn").onclick = async () => {
    const file = document.getElementById("csv-input").files[0];
    if (!file) return alert("Please select a CSV file!");

    document.getElementById("csv-result").innerHTML = "<p class='placeholder-text'>Uploading and analyzing...</p>";

    try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("/predict-csv", {
            method: "POST",
            body: formData
        });
        
        if (!res.ok) {
            throw new Error(`Server error: ${res.statusText}`);
        }

        const data = await res.json();

        let html = "<table><tr><th>Text</th><th>Binary</th><th>Confidence</th><th>Multiclass</th><th>Confidence</th></tr>";
        
        data.results.forEach(r => {
            const binaryClass = getSentimentClass(r.binary);
            const multiClass = getSentimentClass(r.multiclass);
            
            html += `<tr>
                <td>${r.text.length > 50 ? r.text.substring(0, 50) + '...' : r.text}</td>
                <td class="${binaryClass}">${r.binary}</td>
                <td>${(r.binary_conf * 100).toFixed(1)}%</td>
                <td class="${multiClass}">${r.multiclass}</td>
                <td>${(r.multi_conf * 100).toFixed(1)}%</td>
            </tr>`;
        });
        html += "</table>";

        document.getElementById("csv-result").innerHTML = html;

    } catch (error) {
        document.getElementById("csv-result").innerHTML = `<p class="negative">Analysis failed: ${error.message}</p>`;
        console.error("CSV Upload Error:", error);
    }
};