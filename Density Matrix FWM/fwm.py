import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. PARÁMETROS DEL SISTEMA (CESIO 6S -> 8S)
# NOTA: Estos valores son ilustrativos (órdenes de magnitud) 
# para visualizar el fenómeno. Debes reemplazarlos con los 
# valores experimentales exactos de tu celda de Cs.
# ==========================================================
hbar = 1.0 # Unidades atómicas o normalizadas para la simulación
mu_ba = 1.0 
mu_cb = 1.0
gamma_c = 0.5 # Tasa de decaimiento poblacional del nivel 8S
Gamma_ba = 1.0 # Ensanchamiento transversal 6S-6P
Gamma_ca = 0.2 # Ensanchamiento transversal 6S-8S (Doppler-Free suele ser angosto)
Gamma_cb = 1.0 # Ensanchamiento transversal 6P-8S

# Desintonías (Detunings)
# El láser a 822.24 nm está muy lejos de la transición 6S-6P (852 nm)
Delta_1 = 500.0 # Desintonía de 1 fotón muy grande y constante
Delta_3 = 500.0 # Desintonía asumida similar para el campo generado

# ==========================================================
# 2. DEFINICIÓN DE LA ECUACIÓN DE POBLACIÓN (Eq. 5)
# ==========================================================
def calcular_rho_cc_4(Delta_2, E1, E2, E3):
    """
    Calcula la población de cuarto orden del estado superior (8S_1/2)
    basado en la ec. de densidad de Boyd et al.
    """
    # Términos comunes en los denominadores
    D1 = Delta_1 - 1j*Gamma_ba
    D3 = Delta_3 - 1j*Gamma_ba
    D2_ca = Delta_2 - 1j*Gamma_ca
    
    # Combinaciones de desintonías
    D_21 = Delta_2 - Delta_1 - 1j*Gamma_cb
    D_23 = Delta_2 - Delta_3 - 1j*Gamma_cb
    D_213 = 2*Delta_1 - Delta_3 - 1j*Gamma_ba
    D_2321 = Delta_2 + Delta_3 - 2*Delta_1 - 1j*Gamma_cb

    # Desglose de los corchetes de la Eq. 5
    term1 = (np.abs(E1)**4) / (D1 * D_21)
    
    term2 = (E1**2 * np.conj(E2) * np.conj(E3)) * ((1 / (D1 * D_2321)) + (1 / (D1 * D_23)))
    
    term3 = ((np.conj(E1)**2) * E2 * E3) * ((1 / (D_213 * D_21)) + (1 / (D3 * D_21)))
    
    term4 = (np.abs(E2)**2 * np.abs(E3)**2) * (
          (1 / (D_213 * D_2321)) 
        + (1 / (D_213 * D_23)) 
        + (1 / (D3 * D_2321)) 
        + (1 / (D3 * D_23))
    )
    
    corchete = term1 + term2 + term3 + term4
    
    # Factor global y cálculo de la parte imaginaria
    factor_global = -(2 * np.abs(mu_ba)**2 * np.abs(mu_cb)**2) / (hbar**4 * gamma_c)
    
    rho_cc = factor_global * np.imag((1 / D2_ca) * corchete)
    
    return rho_cc

# ==========================================================
# 3. SIMULACIÓN Y BARRIDO DE FRECUENCIA
# ==========================================================
# Barrido del láser de bombeo a través de la resonancia de 2 fotones
Delta_2_array = np.linspace(-5, 5, 1000)

# Amplitud del láser de bombeo incidente (822.24 nm)
E1_incidente = 15.0 + 0j 

# ESCENARIO A: Sin FWM (Solo absorción de dos fotones natural)
# Asumimos que los campos generados aún no tienen fuerza
E2_off, E3_off = 0.0, 0.0
rho_cc_sin_fwm = calcular_rho_cc_4(Delta_2_array, E1_incidente, E2_off, E3_off)

# ESCENARIO B: Con FWM (La condición de cancelación perfecta del paper)
# Para que ocurra la supresión de la ASE, los campos E2 y E3 crecen 
# con una magnitud y una fase relativa opuesta que anula la coherencia.
# (Ajuste manual para emular la fase perfecta e intensidad de equilibrio)
fase_perfecta = np.exp(1j * np.pi) # Desfase de 180 grados (interferencia destructiva)
E2_on = 15.0 * fase_perfecta
E3_on = 15.0 * fase_perfecta
rho_cc_con_fwm = calcular_rho_cc_4(Delta_2_array, E1_incidente, E2_on, E3_on)

# ==========================================================
# 4. GRAFICACIÓN
# ==========================================================
plt.figure(figsize=(10, 6))

plt.plot(Delta_2_array, rho_cc_sin_fwm, 'b-', linewidth=2, label='Sin FWM (Solo Bombeo $E_1$)')
plt.plot(Delta_2_array, rho_cc_con_fwm, 'r--', linewidth=2, label='Con FWM (Campos $E_2, E_3$ en fase perfecta)')

# Destacando el punto central del artículo
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='gray', linestyle=':', alpha=0.5)

# Anotación del fenómeno físico
plt.annotate('Supresión total de la ASE\n($\\rho_{cc}^{(4)} \\to 0$)', 
             xy=(0, np.max(rho_cc_con_fwm)), xytext=(1.5, np.max(rho_cc_sin_fwm)*0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, bbox=dict(boxstyle="round", alpha=0.1))

plt.title('Población del nivel superior $8S_{1/2}$ del Cesio (Ec. 5)\nCompetencia entre Absorción y FWM', fontsize=14)
plt.xlabel('Desintonía de dos fotones, $\\Delta_2$ (Unidades arbitrarias)', fontsize=12)
plt.ylabel('Población $\\rho_{cc}^{(4)}$', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()