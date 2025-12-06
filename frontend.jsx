import React, { useState } from 'react';
import { AlertCircle, CheckCircle2, XCircle, Loader } from 'lucide-react';

const TelcoChurnPredictionApp = () => {
  const [formData, setFormData] = useState({
    tenure: '',
    MonthlyCharges: '',
    TotalCharges: '',
    TechSupport_yes: 0,
    Contract_one_year: 0,
    Contract_two_year: 0,
    Partner_yes: 0,
    StreamingTV_yes: 0,
    StreamingTV_no_internet_service: 0,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE_URL = 'http://localhost:8000';

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    if (type === 'checkbox') {
      setFormData(prev => ({
        ...prev,
        [name]: checked ? 1 : 0
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Validate numeric inputs
    if (!formData.tenure || !formData.MonthlyCharges || !formData.TotalCharges) {
      setError('Please fill in all numeric fields');
      setLoading(false);
      return;
    }

    try {
      const payload = {
        tenure: parseFloat(formData.tenure),
        MonthlyCharges: parseFloat(formData.MonthlyCharges),
        TechSupport_yes: parseInt(formData.TechSupport_yes),
        Contract_one_year: parseInt(formData.Contract_one_year),
        Contract_two_year: parseInt(formData.Contract_two_year),
        TotalCharges: parseFloat(formData.TotalCharges),
        Partner_yes: parseInt(formData.Partner_yes),
        StreamingTV_yes: parseInt(formData.StreamingTV_yes),
        StreamingTV_no_internet_service: parseInt(formData.StreamingTV_no_internet_service),
      };

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to get prediction. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      tenure: '',
      MonthlyCharges: '',
      TotalCharges: '',
      TechSupport_yes: 0,
      Contract_one_year: 0,
      Contract_two_year: 0,
      Partner_yes: 0,
      StreamingTV_yes: 0,
      StreamingTV_no_internet_service: 0,
    });
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Telco Customer Churn Prediction
          </h1>
          <p className="text-gray-600">
            Enter customer details to predict churn probability
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Form Section */}
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              Customer Information
            </h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Numeric Inputs */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tenure (months) *
                </label>
                <input
                  type="number"
                  name="tenure"
                  value={formData.tenure}
                  onChange={handleInputChange}
                  min="0"
                  step="0.1"
                  placeholder="e.g., 24"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Monthly Charges ($) *
                </label>
                <input
                  type="number"
                  name="MonthlyCharges"
                  value={formData.MonthlyCharges}
                  onChange={handleInputChange}
                  min="0"
                  step="0.01"
                  placeholder="e.g., 65.00"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Total Charges ($) *
                </label>
                <input
                  type="number"
                  name="TotalCharges"
                  value={formData.TotalCharges}
                  onChange={handleInputChange}
                  min="0"
                  step="0.01"
                  placeholder="e.g., 1500.00"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Boolean Inputs */}
              <div className="pt-4 border-t border-gray-200">
                <h3 className="text-sm font-medium text-gray-700 mb-4">Services & Contract</h3>
                
                <div className="space-y-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      name="TechSupport_yes"
                      checked={formData.TechSupport_yes === 1}
                      onChange={handleInputChange}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <span className="ml-3 text-gray-700">Has Tech Support</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      name="Partner_yes"
                      checked={formData.Partner_yes === 1}
                      onChange={handleInputChange}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <span className="ml-3 text-gray-700">Has Partner</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      name="StreamingTV_yes"
                      checked={formData.StreamingTV_yes === 1}
                      onChange={handleInputChange}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <span className="ml-3 text-gray-700">Streaming TV Service</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      name="StreamingTV_no_internet_service"
                      checked={formData.StreamingTV_no_internet_service === 1}
                      onChange={handleInputChange}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <span className="ml-3 text-gray-700">No Internet Service</span>
                  </label>
                </div>
              </div>

              {/* Contract Section */}
              <div className="pt-4 border-t border-gray-200">
                <h3 className="text-sm font-medium text-gray-700 mb-4">Contract Type</h3>
                
                <div className="space-y-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      name="Contract_one_year"
                      checked={formData.Contract_one_year === 1}
                      onChange={handleInputChange}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <span className="ml-3 text-gray-700">1-Year Contract</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      name="Contract_two_year"
                      checked={formData.Contract_two_year === 1}
                      onChange={handleInputChange}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <span className="ml-3 text-gray-700">2-Year Contract</span>
                  </label>
                </div>
              </div>

              {/* Buttons */}
              <div className="flex gap-4 pt-6">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-2 px-4 rounded-lg transition"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <Loader className="w-4 h-4 animate-spin" />
                      Predicting...
                    </span>
                  ) : (
                    'Get Prediction'
                  )}
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-800 font-semibold py-2 px-4 rounded-lg transition"
                >
                  Reset
                </button>
              </div>
            </form>
          </div>

          {/* Results Section */}
          <div className="lg:sticky lg:top-8 h-fit">
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-red-800">Error</h3>
                    <p className="text-red-700 text-sm mt-1">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {result && !error && (
              <div className="bg-white rounded-lg shadow-lg p-8">
                <h2 className="text-2xl font-semibold text-gray-800 mb-6">
                  Prediction Result
                </h2>

                {/* Churn Status Card */}
                <div className={`rounded-lg p-6 mb-6 ${
                  result.churn_prediction === 'Yes'
                    ? 'bg-red-50 border border-red-200'
                    : 'bg-green-50 border border-green-200'
                }`}>
                  <div className="flex items-center gap-3 mb-3">
                    {result.churn_prediction === 'Yes' ? (
                      <XCircle className="w-8 h-8 text-red-600" />
                    ) : (
                      <CheckCircle2 className="w-8 h-8 text-green-600" />
                    )}
                    <span className={`text-lg font-semibold ${
                      result.churn_prediction === 'Yes'
                        ? 'text-red-800'
                        : 'text-green-800'
                    }`}>
                      Prediction: {result.churn_prediction}
                    </span>
                  </div>
                  <p className={`text-sm ${
                    result.churn_prediction === 'Yes'
                      ? 'text-red-700'
                      : 'text-green-700'
                  }`}>
                    {result.churn_prediction === 'Yes'
                      ? 'This customer is likely to churn. Consider retention strategies.'
                      : 'This customer is likely to stay. Focus on maintaining satisfaction.'}
                  </p>
                </div>

                {/* Metrics */}
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600 mb-2">Churn Probability</p>
                    <p className="text-3xl font-bold text-gray-800">
                      {(result.churn_probability * 100).toFixed(2)}%
                    </p>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600 mb-2">Model Confidence</p>
                    <p className="text-3xl font-bold text-gray-800">
                      {(result.confidence * 100).toFixed(2)}%
                    </p>
                  </div>

                  {/* Progress Bar */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600 mb-3">Probability Distribution</p>
                    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          result.churn_probability > 0.5
                            ? 'bg-red-500'
                            : 'bg-green-500'
                        }`}
                        style={{ width: `${result.churn_probability * 100}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-gray-600 mt-2">
                      <span>No Churn</span>
                      <span>Churn</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!result && !error && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                <p className="text-blue-800 text-center">
                  Fill in the customer details and click "Get Prediction" to see results
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TelcoChurnPredictionApp;