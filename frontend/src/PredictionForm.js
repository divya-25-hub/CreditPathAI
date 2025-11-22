import React, { useState } from "react";
import { predictRisk } from "./api";

function PredictionForm() {
  const [formData, setFormData] = useState({
    income: "",
    loan_amount: "",
    credit_score: "",
    rate_of_interest: "",
    ltv: "",
    dtir1: "",
  });

  const [result, setResult] = useState(null);

  // Handle input changes
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  // Submit to API
  const handleSubmit = async (e) => {
    e.preventDefault();

    const payload = {
      income: Number(formData.income),
      loan_amount: Number(formData.loan_amount),
      credit_score: Number(formData.credit_score),
      rate_of_interest: Number(formData.rate_of_interest),
      ltv: Number(formData.ltv),
      dtir1: Number(formData.dtir1),
    };

    const response = await predictRisk(payload);
    setResult(response);
  };

  return (
    <div style={{ maxWidth: "400px", margin: "auto" }}>
      <h2>Loan Risk Prediction</h2>

      <form onSubmit={handleSubmit}>

        <input
          type="number"
          name="income"
          placeholder="Income"
          value={formData.income}
          onChange={handleChange}
          required
        />

        <input
          type="number"
          name="loan_amount"
          placeholder="Loan Amount"
          value={formData.loan_amount}
          onChange={handleChange}
          required
        />

        <input
          type="number"
          name="credit_score"
          placeholder="Credit Score"
          value={formData.credit_score}
          onChange={handleChange}
          required
        />

        <input
          type="number"
          name="rate_of_interest"
          placeholder="Rate of Interest"
          value={formData.rate_of_interest}
          onChange={handleChange}
          required
        />

        <input
          type="number"
          name="ltv"
          placeholder="LTV Ratio"
          value={formData.ltv}
          onChange={handleChange}
          required
        />

        <input
          type="number"
          name="dtir1"
          placeholder="DTI Ratio"
          value={formData.dtir1}
          onChange={handleChange}
          required
        />

        <button type="submit">Predict</button>
      </form>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Risk Category: {result.risk_category}</h3>
          <p>Default Probability: {result.probability}</p>
          <p>Recommended Action: {result.recommendation}</p>
        </div>
      )}
    </div>
  );
}

export default PredictionForm;
