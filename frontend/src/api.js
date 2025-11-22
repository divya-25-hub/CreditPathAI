export async function predictRisk(data) {
  const response = await fetch(
    "https://legendary-space-sniffle-r7g5pvq6j552p56q-8000.app.github.dev /predict",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }
  );

  if (!response.ok) {
    console.error("API Error:", response.statusText);
    throw new Error("Failed to fetch");
  }

  return response.json();
}
