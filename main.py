import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm, uniform, expon

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

def plot_prior_posterior(prior, likelihood):
    # Sample data for plotting
    data = np.linspace(0, 1, 1000)  # Adjust based on the distribution support

    # Prior distribution
    if prior == "Beta":
        prior_dist = beta.pdf(data, 2, 2)  # Adjust shape parameters
    elif prior == "Gamma":
        prior_dist = gamma.pdf(data, 2, scale=1)  # Adjust shape and scale parameters
    elif prior == "Normal":
        prior_dist = norm.pdf(data, loc=0, scale=1)  # Adjust mean and standard deviation
    elif prior == "Uniform":
        prior_dist = uniform.pdf(data)  # No parameters needed for uniform
    elif prior == "Exponential":
        prior_dist = expon.pdf(data, scale=1)  # Adjust scale parameter

    # Likelihood distribution
    # For simplicity, assuming likelihood has the same parameters in each case
    if likelihood == "Bernoulli":
        likelihood_dist = np.random.choice([0, 1], size=len(data), p=[0.3, 0.7])
    elif likelihood == "Poisson":
        likelihood_dist = np.random.poisson(5, size=len(data))
    elif likelihood == "Normal":
        likelihood_dist = np.random.normal(loc=0, scale=1, size=len(data))

    # Update prior based on observed data (posterior distribution)
    if prior == "Beta" and likelihood == "Bernoulli":
        posterior_dist = beta.pdf(data, 2 + np.sum(likelihood_dist), 2 + len(likelihood_dist) - np.sum(likelihood_dist))
    elif prior == "Gamma" and likelihood == "Poisson":
        posterior_dist = gamma.pdf(data, 2 + np.sum(likelihood_dist), scale=1 / (1 + len(likelihood_dist)))
    elif prior == "Normal" and likelihood == "Normal":
        posterior_dist = norm.pdf(data, loc=np.mean(likelihood_dist), scale=1)  # Assuming known variance
    else:
        # Non-conjugate case - posterior not analytically tractable
        posterior_dist = np.ones_like(data)  # Placeholder for non-conjugate case

    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    ax[0].plot(data, prior_dist, label='Prior')
    ax[0].set_title('Prior Distribution')
    ax[1].plot(data, likelihood_dist, label='Likelihood')
    ax[1].set_title('Likelihood Distribution')
    ax[2].plot(data, posterior_dist, label='Posterior')
    ax[2].set_title('Posterior Distribution')

    for a in ax:
        a.legend()
        a.grid(True)

    st.pyplot(fig)

def main():
    st.title("Conjugate and Non-Conjugate Priors Information with Visualizations")

    # Dropdowns for prior and likelihood
    prior = st.selectbox("Select Prior Distribution", ["Beta", "Gamma", "Normal", "Uniform", "Exponential"])
    likelihood = st.selectbox("Select Likelihood Distribution", ["Bernoulli", "Poisson", "Normal"])

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
    plot_prior_posterior(prior, likelihood)

if __name__ == "__main__":
    main()