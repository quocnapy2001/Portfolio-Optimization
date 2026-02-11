# Portfolio-Optimization

### Assets:
<img width="470" height="286" alt="image" src="https://github.com/user-attachments/assets/7c458e75-36cc-4d4e-9929-bcbb80b632ad" />

### Static Scenario: 
<img width="568" height="280" alt="image" src="https://github.com/user-attachments/assets/0388867a-0d1b-4c9a-8f69-abe419d631fd" />

### Unconstrained Optimization (Base):
- Rebalancing frequency: monthly
- Lookback window for optimisation: 3 years
- Portfolio weights held constant between rebalancing dates

<img width="720" height="399" alt="image" src="https://github.com/user-attachments/assets/a9292cc1-d7a2-4a4d-a0ff-4f91062be80e" />
<img width="609" height="278" alt="image" src="https://github.com/user-attachments/assets/15ce1379-6c8b-45e5-a8fc-deb1430a6c2f" />
<img width="712" height="424" alt="image" src="https://github.com/user-attachments/assets/ad086dbb-2ab2-4ea4-a7b4-2b955e69c3fb" />

- Portfolio turnover is quantified using the L1 norm of changes in portfolio weights, computed as the sum of absolute differences between consecutive rebalancing allocations. This metric captures the total proportion of the portfolio reallocated at each rebalance and is consistent with linear transaction-cost assumptions, providing a transparent measure of trading intensity in a long-only portfolio framework.

### Constrained Optimization:
- Minimum weight per asset: 5%
- Maximum weight per asset: 20%
- Long-only constraint applied (no short positions)
- Regularisation method: L2 regularisation
- Regularisation strength (gamma): 0.1
- Purpose of regularisation: reduce extreme weights and improve stability


<img width="720" height="399" alt="image" src="https://github.com/user-attachments/assets/31d4695e-0434-4256-9352-635cf982a9af" />
<img width="615" height="278" alt="image" src="https://github.com/user-attachments/assets/0788d745-d8cd-4bb2-b77a-9668d23a8f22" />
<img width="712" height="424" alt="image" src="https://github.com/user-attachments/assets/2f2a9208-aae4-4c3a-905b-08b6c49832bf" />







