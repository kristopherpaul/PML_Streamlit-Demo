import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm, uniform, expon, poisson, bernoulli

def get_prior_params(prior):
    if prior == "Beta":
        param_a = st.number_input("Enter Beta Parameter a", min_value=0.01, value=1.0)
        param_b = st.number_input("Enter Beta Parameter b", min_value=0.01, value=1.0)
        return [param_a, param_b]
    elif prior == "Gamma":
        param_shape = st.number_input("Enter Gamma Shape Parameter", min_value=0.01, value=1.0)
        param_scale = st.number_input("Enter Gamma Scale Parameter", min_value=0.01, value=1.0)
        return [param_shape, param_scale]
    elif prior == "Normal":
        param_mean = st.number_input("Enter Normal Mean Parameter", value=0.0)
        param_std = st.number_input("Enter Normal Standard Deviation Parameter", min_value=0.01, value=1.0)
        return [param_mean, param_std]
    elif prior == "Uniform":
        # No parameters needed for uniform
        return []
    elif prior == "Exponential":
        param_scale = st.number_input("Enter Exponential Scale Parameter", min_value=0.01, value=1.0)
        return [param_scale]

def get_likelihood_params(likelihood):
    # Add more cases for different likelihood distributions
    if likelihood == "Bernoulli":
        param_p = st.number_input("Enter Bernoulli Parameter p", min_value=0.01, max_value=0.99, value=0.5)
        return [param_p]
    elif likelihood == "Poisson":
        param_lambda = st.number_input("Enter Poisson Parameter lambda", min_value=0.01, value=1.0)
        return [param_lambda]
    elif likelihood == "Normal":
        param_mean = st.number_input("Enter Normal Mean Parameter", value=0.0)
        param_std = st.number_input("Enter Normal Standard Deviation Parameter", min_value=0.01, value=1.0)
        return [param_mean, param_std]

def conjugate_prior_info(prior, likelihood):
    if prior == "Beta" and likelihood == "Bernoulli":
        return "No transformation needed", "Beta (Conjugate)"
    elif prior == "Gamma" and likelihood == "Poisson":
        return "No transformation needed", "Gamma (Conjugate)"
    elif prior == "Normal" and likelihood == "Normal":
        return "No transformation needed", "Normal (Conjugate)"
    else:
        return "Unknown", "Unknown (Non-Conjugate)"

def non_conjugate_prior_info(prior, likelihood):
    if prior == "Normal" and likelihood == "Bernoulli":
        return "Apply Sigmoid function to the parameter", "Unknown (Non-Conjugate)"
    elif prior == "Uniform" and likelihood == "Poisson":
        return "No transformation needed", "Unknown (Non-Conjugate)"
    elif prior == "Exponential" and likelihood == "Normal":
        return "No transformation needed", "Unknown (Non-Conjugate)"
    else:
        return "Unknown", "Unknown (Non-Conjugate)"

def plot_prior_posterior(prior, likelihood, prior_params, likelihood_params, num_likelihood_samples):
    # Sample data for plotting
    data = np.linspace(0, 1, 1000)  # Adjust based on the distribution support

    # Use user-provided parameters for the prior distribution
    if prior == "Beta":
        prior_dist = beta.pdf(data, prior_params[0], prior_params[1])  # Adjust shape parameters
    elif prior == "Gamma":
        prior_dist = gamma.pdf(data, prior_params[0], scale=prior_params[1])  # Adjust shape and scale parameters
    elif prior == "Normal":
        prior_dist = norm.pdf(data, loc=prior_params[0], scale=prior_params[1])  # Adjust mean and standard deviation
    elif prior == "Uniform":
        prior_dist = uniform.pdf(data)  # No parameters needed for uniform
    elif prior == "Exponential":
        prior_dist = expon.pdf(data, scale=prior_params[0])  # Adjust scale parameter

    # Use user-provided parameters for the likelihood distribution
    if likelihood == "Bernoulli":
        likelihood_dist = bernoulli.rvs(likelihood_params[0], size=num_likelihood_samples)
    elif likelihood == "Poisson":
        likelihood_dist = poisson.rvs(likelihood_params[0], size=num_likelihood_samples)
    elif likelihood == "Normal":
        likelihood_dist = norm.rvs(loc=likelihood_params[0], scale=likelihood_params[1], size=num_likelihood_samples)
    
    # Update prior based on observed data (posterior distribution)
    if prior == "Beta" and likelihood == "Bernoulli":
        posterior_dist = beta.pdf(data, prior_params[0] + (likelihood_dist==1).sum(), prior_params[1] + (likelihood_dist==0).sum())
    elif prior == "Gamma" and likelihood == "Poisson":
        posterior_dist = gamma.pdf(data, prior_params[0] + (likelihood_dist==1).sum(), scale=1/(prior_params[1]+num_likelihood_samples))
    elif prior == "Normal" and likelihood == "Normal":
        posterior_mean = (prior_params[0]/prior_params[1]**2 + likelihood_params[0]/likelihood_params[1]**2) / (1/prior_params[1]**2 + 1/likelihood_params[1]**2)
        posterior_std_dev = np.sqrt(1/(1/prior_params[1]**2 + 1/likelihood_params[1]**2))
        posterior_dist = norm.pdf(data, loc=posterior_mean, scale=posterior_std_dev)
    else:
        # Non-conjugate case - posterior not analytically tractable
        posterior_dist = np.ones_like(data)  # Placeholder for non-conjugate case

    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    ax[0].plot(data, prior_dist, label='Prior')
    ax[0].set_title('Prior Distribution')
    ax[1].hist(likelihood_dist, label='Likelihood')
    ax[1].set_title('Likelihood Distribution')
    ax[2].plot(data, posterior_dist, label='Posterior')
    ax[2].set_title('Posterior Distribution')

    for a in ax:
        #a.legend()
        a.grid(True)

    st.pyplot(fig)

def main():
    st.title("Conjugate and Non-Conjugate Priors Information with Visualizations")

    # Dropdowns for prior and likelihood
    prior = st.selectbox("Select Prior Distribution", ["Beta", "Gamma", "Normal", "Uniform", "Exponential"])
    likelihood = st.selectbox("Select Likelihood Distribution", ["Bernoulli", "Poisson", "Normal"])

    # Take user input for prior parameters
    prior_params = get_prior_params(prior)

    # Take user input for likelihood parameters
    likelihood_params = get_likelihood_params(likelihood)
    
    # Take user input for the number of likelihood samples
    num_likelihood_samples = st.number_input("Enter the number of Likelihood Samples", min_value=1, value=100)

    # Display information based on choices
    if prior in ["Beta", "Gamma", "Normal"] and likelihood in ["Bernoulli", "Poisson", "Normal"]:
        bijector, posterior_type = conjugate_prior_info(prior, likelihood)
    else:
        bijector, posterior_type = non_conjugate_prior_info(prior, likelihood)

    # Display results
    st.subheader("Results:")
    st.write(f"Bijector (Parameter Transform): {bijector}")
    st.write(f"Posterior Distribution Type: {posterior_type}")

    # Visualizations
    st.subheader("Visualizations:")
    plot_prior_posterior(prior, likelihood, prior_params, likelihood_params, num_likelihood_samples)

if __name__ == "__main__":
    main()