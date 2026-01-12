document.getElementById('analyze-btn').addEventListener('click', async () => {
    // 1. Get references to the HTML elements where we want to show data
    const geminiDiv = document.getElementById('gemini-output');
    const modelDiv = document.getElementById('model-output');
    const factText = document.getElementById('random-fact');

    // 2. Set "Loading" states so the user knows the AI is working
    geminiDiv.innerHTML = "<p class='placeholder-text'>Consulting Gemini AI Marine Expert...</p>";
    modelDiv.innerHTML = "<p class='placeholder-text'>Running RGB Analysis...</p>";
    factText.innerText = "Fetching a new reef fact...";

    try {
        // 3. FETCH THE FACTS (GET request to your Flask /get_facts route)
        const factResponse = await fetch('/get_facts');
        const facts = await factResponse.json();
        
        // Update the "Did you know?" box with the first fact description
        // We use facts[0] because your Python returns a list
        if (facts && facts.length > 0) {
            factText.innerText = facts[0].description;
        }

        // 4. FETCH THE ANALYSIS (POST request to your Flask /analyze route)
        const analysisResponse = await fetch('/analyze', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const analysis = await analysisResponse.json();
        
        // 5. UPDATE THE UI with the results from Python
        modelDiv.innerHTML = `<p>${analysis.model}</p>`;
        geminiDiv.innerHTML = `<p>${analysis.gemini}</p>`;
        
    } catch (error) {
        console.error("Error connecting to backend:", error);
        geminiDiv.innerHTML = "<p style='color: red;'>Error: Could not reach the AI server.</p>";
        modelDiv.innerHTML = "<p style='color: red;'>Error: Backend offline.</p>";
        factText.innerText = "Check your Flask server!";
    }
});