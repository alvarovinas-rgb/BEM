import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ======================================================
# CONFIGURACIÓ DE LA PÀGINA
# ======================================================
st.set_page_config(page_title="BEM Solver - UPC", layout="wide")

st.title("🛸 Motor de Càlcul BEM (Blade Element Momentum)")
st.subheader("Entorn de simulació per a la validació de turbines")

# ======================================================
# INTERFÍCIE DE L'USUARI (BARRA LATERAL)
# ======================================================
st.sidebar.header("⚙️ Paràmetres del Simulador")

# --- Fluids i Rotor de base ---
density_f = 1.225        # kg/m³, densidad del fluido
radius_d = 126 / 2       # m, radio total del rotor
velocity_ang = 1.2671    # rad/s, velocidad angular
N = 3

# --- Controls Interactius per al TSR ---
st.sidebar.subheader("📊 Configuració del TSR")
starter_TSR = st.sidebar.number_input("TSR Inicial", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
finish_TSR = st.sidebar.number_input("TSR Final", min_value=0.5, max_value=20.0, value=10.0, step=0.5)
discretitzacio_TSR = st.sidebar.number_input("TSR Delta (Pas)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

# --- Controls Interactius per al Pitch ---
st.sidebar.subheader("📐 Configuració del Pitch")
starter_pitch = st.sidebar.number_input("Pitch Inicial (º)", min_value=-5.0, max_value=30.0, value=0.0, step=0.5)
finish_pitch = st.sidebar.number_input("Pitch Final (º)", min_value=-5.0, max_value=30.0, value=2.0, step=0.5)
discretitzacio_pitch = st.sidebar.number_input("Pitch Delta (Pas)", min_value=0.1, max_value=10.0, value=1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🔢 Paràmetres Numèrics")
max_iter = st.sidebar.number_input("Màxim d'iteracions", min_value=10, max_value=500, value=100, step=10)
tol = st.sidebar.number_input("Precisió (Tolerància)", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.5f")
omega = st.sidebar.slider("Factor de relaxació ($\omega$)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

# ======================================================
# RUTES DELS FITXERS EXCEL
# ======================================================
directori_actual = os.path.dirname(os.path.abspath(__file__))

ruta = os.path.join(directori_actual, "Turbines", "NREL_5MW", "parametros_foil.xlsx")
ruta_aero_data = os.path.join(directori_actual, "Turbines", "AeroData", "AeroData.xlsx")

# ======================================================
# BOTÓ D'EXECUCIÓ PRINCIPAL
# ======================================================
if st.button("▶️ Executar Simulació BEM", type="primary"):
    
    with st.spinner("Simulant el comportament aerodinàmic..."):
        try:
            # 1. Carregar dades dels Excels
            df_NREL_5MW = pd.read_excel(ruta, sheet_name='TSR=7', usecols=range(8))
            df_aerodata = pd.read_excel(ruta_aero_data, sheet_name=None)
            
            # Generar els arrays a partir de les dades del formulari
            TSR = np.arange(starter_TSR, finish_TSR + discretitzacio_TSR, discretitzacio_TSR)
            pitch = np.arange(starter_pitch, finish_pitch + discretitzacio_pitch, discretitzacio_pitch)
            velocity_inf_list = (velocity_ang * radius_d) / TSR 
            
            Cp_geometric_ideal_list = np.zeros_like(velocity_inf_list, dtype=float)
            Cp_real_list = np.zeros_like(velocity_inf_list, dtype=float)
            results_pitch_list = []

            # Bucle de càlcul (El teu motor BEM)
            for p in pitch:
                for j, velocity_inf_f in enumerate(velocity_inf_list):
                    
                    specific_radius = df_NREL_5MW.iloc[:, 0].to_numpy()
                    chord = df_NREL_5MW.iloc[:, 1].to_numpy()
                    twist = df_NREL_5MW.iloc[:, 2].to_numpy()
                    perfiles = df_NREL_5MW.iloc[:, 4].to_numpy()

                    divisions = len(specific_radius)
                    dr_d = radius_d / divisions
                    
                    factor_axial = np.zeros(divisions)
                    factor_tangencial = np.zeros(divisions)
                    c_lift = np.zeros(divisions)
                    c_drag = np.zeros(divisions)

                    beta = twist + p
                    local_solidity = np.array([(N * c) / (2 * np.pi * r) for c, r in zip(chord, specific_radius)])

                    # Bucle iteratiu d'inducció
                    for iter_idx in range(max_iter):
                        v_axial_blade = velocity_inf_f * (1 - factor_axial)
                        v_tan_air_blade = velocity_ang * specific_radius * (1 + factor_tangencial)

                        phi = np.arctan2(v_axial_blade, v_tan_air_blade)
                        sin_phi = np.maximum(np.sin(phi), 0.01)
                        cos_phi = np.cos(phi)

                        alpha_deg = np.rad2deg(phi - np.deg2rad(beta))

                        for i, perfil in enumerate(perfiles):
                            df_polar = df_aerodata[perfil]
                            c_lift[i] = np.interp(alpha_deg[i], df_polar['Alpha'], df_polar['C_lift'])
                            c_drag[i] = np.interp(alpha_deg[i], df_polar['Alpha'], df_polar['C_drag'])

                        Cx = c_lift * cos_phi + c_drag * sin_phi
                        Cy = c_lift * sin_phi - c_drag * cos_phi

                        f_tip = (N / 2.0) * (radius_d - specific_radius) / (specific_radius * sin_phi)
                        F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), 0, 1))
                        F = np.where(F < 0.01, 0.01, F) 

                        CT = (local_solidity * Cx * (1 - factor_axial)**2) / (sin_phi**2)
                        new_factor_axial = np.zeros_like(factor_axial)
                        
                        mask_bem = (CT <= 0.96) 
                        new_factor_axial[mask_bem] = 1 / ((4 * F[mask_bem] * sin_phi[mask_bem]**2) / (local_solidity[mask_bem] * Cx[mask_bem]) + 1)
                        mask_glauert = ~mask_bem
                        new_factor_axial[mask_glauert] = 0.143 + np.sqrt(np.maximum(0, 0.0203 - 0.6427 * (0.889 - CT[mask_glauert])))

                        new_factor_tangencial = 1 / ((4 * F * sin_phi * cos_phi) / (local_solidity * Cy) - 1)

                        old_t = factor_tangencial.copy()
                        old_a = factor_axial.copy()
                        factor_axial = old_a + omega * (new_factor_axial - old_a)
                        factor_tangencial = factor_tangencial + omega * (new_factor_tangencial - factor_tangencial)

                        error_axial = np.max(np.abs(factor_axial - old_a))
                        error_tan = np.max(np.abs(factor_tangencial - old_t))

                        if error_axial < tol and error_tan < tol:
                            break

                    # Forces i Potència
                    v_axial_air_final = velocity_inf_f * (1 - factor_axial)
                    v_tan_air_final = velocity_ang * specific_radius * (1 + factor_tangencial)
                    W = np.sqrt(v_axial_air_final**2 + v_tan_air_final**2)

                    C_y = c_lift * np.sin(phi) - c_drag * np.cos(phi)
                    C_x = c_lift * np.cos(phi) + c_drag * np.sin(phi)

                    dF_theta = (0.5 * density_f * W**2 * chord * C_y) * dr_d
                    dQ = specific_radius * dF_theta

                    power_shaft_total = (N * velocity_ang * np.sum(dQ)) / 1e3
                    A_rotor = np.pi * radius_d**2
                    P_wind = (0.5 * density_f * A_rotor * velocity_inf_f**3) / 1e3

                    Cp_ideal = (2 / radius_d**2) * np.trapz(4 * factor_axial * (1 - factor_axial)**2 * specific_radius, specific_radius)
                    Cp_real = power_shaft_total / P_wind

                    Cp_geometric_ideal_list[j] = Cp_ideal
                    Cp_real_list[j] = Cp_real

                results_df = pd.DataFrame({"TSR": TSR, "Cp_ideal": Cp_geometric_ideal_list.copy(), "Cp_real": Cp_real_list.copy(), "Pitch": p})
                results_pitch_list.append(results_df)

            st.success("🎯 Càlcul completat de manera correcta!")

            # ======================================================
            # PROCESSAT DE RESULTATS I GRÀFICS
            # ======================================================
            df_total = pd.concat(results_pitch_list, ignore_index=True)
            df_optimo = df_total.loc[df_total.groupby('TSR')['Cp_real'].idxmax()].sort_values('TSR')

            # Organitzem els resultats visuals en columnes
            col_graf1, col_graf2 = st.columns(2)

            with col_graf1:
                st.write("### 📈 Gràfica Rendiment (Real vs Ideal)")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                for df in results_pitch_list:
                    p_actual = df["Pitch"].iloc[0]
                    ax1.plot(df["TSR"], df["Cp_real"], marker='o', markersize=4, label=f"Pitch {p_actual}º")
                    if p_actual == 0:
                        ax1.plot(df["TSR"], df["Cp_ideal"], color='black', linestyle='--', label="CP geometric ideal")

                ax1.axhline(y=0.593, color='red', linestyle='--', alpha=0.6, label="Límite de Betz")
                ax1.set_xlabel("TSR")
                ax1.set_ylabel("Cp")
                ax1.set_ylim(0, 0.65)
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)

            with col_graf2:
                st.write("### 💎 Límit d'Operació Òptima")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.plot(df_optimo["TSR"], df_optimo["Cp_real"], color='blue', marker='s', label="Cp óptimo")
                
                for i, row in df_optimo.iterrows():
                    ax2.annotate(f"{row['Pitch']}º", (row['TSR'], row['Cp_real']), textcoords="offset points", xytext=(0,10), ha='center')

                ax2.set_xlabel("TSR")
                ax2.set_ylabel("Cp")
                ax2.set_ylim(0, 0.6)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                st.pyplot(fig2)

            # Taula final a sota de tot
            st.divider()
            st.write("### 📋 Valors Òptims calculats de TSR i Pitch")
            st.dataframe(df_optimo[['TSR', 'Pitch', 'Cp_real']].reset_index(drop=True), use_container_width=True)

        except Exception as e:
            st.error(f"S'ha produït un error durant l'execució: {e}")
            st.info("Revisa que els fitxers Excel estiguin tancats i que les rutes definides al codi siguin les correctes.")
else:
    st.info("💡 Modifica els paràmetres a la barra de l'esquerra i clica al botó per començar la simulació.")