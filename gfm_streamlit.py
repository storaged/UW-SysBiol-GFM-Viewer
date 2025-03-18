import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import re
from datetime import datetime

############################
# GA Simulation Components #
############################

def fitness_function(phenotypes, optimum, alpha):
    """
    Geometric Fisher Model-like fitness:
    w(x) = exp(-alpha * distance^2),
    where distance is the Euclidean distance from the current optimum.
    """
    diff = phenotypes - optimum  # shape: (N, 2)
    dist_sq = np.sum(diff**2, axis=1)  # shape: (N,)
    fitness_values = np.exp(-alpha * dist_sq)
    return fitness_values

def select_and_reproduce(phenotypes, fitnesses, N):
    """
    Given phenotypes and their fitnesses, produce the next generation.
    We sample indices proportional to fitness, then clone those phenotypes.
    Also track how many offspring each individual got (to enable diagnostics).
    """
    fitness_sum = np.sum(fitnesses)
    if fitness_sum == 0:
        prob = np.ones_like(fitnesses) / len(fitnesses)
    else:
        prob = fitnesses / fitness_sum
    chosen_indices = np.random.choice(len(phenotypes), size=N, p=prob)
    
    # Count offspring for each individual
    offspring_counts = np.bincount(chosen_indices, minlength=len(phenotypes))

    next_gen = phenotypes[chosen_indices]
    return next_gen, offspring_counts

def mutate(phenotypes, mutation_std):
    """
    Add Gaussian noise with std=mutation_std to each phenotype (in 2D).
    """
    mutations = np.random.normal(loc=0.0, scale=mutation_std, size=phenotypes.shape)
    return phenotypes + mutations

##########################
# Simulation Main Method #
##########################

def run_simulation(
    N,
    T,
    alpha,
    mutation_std,
    shift_per_step,
    frames_dir="frames",
    global_view=False,
    create_diag_plots=False,
    progress_bar=None
):
    """
    Run the genetic algorithm simulation with Geometric Fisher Model fitness
    in 2D, store frames, and produce:
      1) A main evolution GIF
      2) Optionally, a second "diagnostic" GIF with time-series data.

    Parameters
    ----------
    N : int
        Population size
    T : int
        Number of time steps (generations)
    alpha : float
        Selection strength
    mutation_std : float
        Standard deviation for mutation
    shift_per_step : (float, float)
        Amount to shift optimum each generation (dx, dy)
    frames_dir : str
        Directory to store frames and final GIF
    global_view : bool
        If True, gather all simulation data first, then compute a single global
        bounding box for all frames. Otherwise, each frame is scaled dynamically.
    create_diag_plots : bool
        If True, generate a second GIF showing diagnostic plots over time.
    progress_bar : streamlit.progress (optional)
        If provided, will be used to update progress.

    Returns
    -------
    (main_gif_path, diag_gif_path or None)
       Returns paths to the main GIF and the diagnostic GIF (None if not created).
    """

    # Create frames directory if it doesn't exist
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Timestamp for unique file naming
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build GIF filename with parameters (rounded to 3 decimals)
    base_gif_filename = (
        f"ga_simulation_{N}_{T}_"
        f"{alpha:.3f}_"
        f"{mutation_std:.3f}_"
        f"{shift_per_step[0]:.3f}_"
        f"{shift_per_step[1]:.3f}_"
        f"{timestamp_str}"
    )
    main_gif_filename = base_gif_filename + ".gif"
    main_gif_path = os.path.join(frames_dir, main_gif_filename)

    diag_gif_path = None
    if create_diag_plots:
        # We'll store a second GIF for diagnostics
        diag_gif_path = os.path.join(frames_dir, "diag_" + main_gif_filename)

    # Initialize population in 2D
    phenotypes = np.random.normal(loc=0.0, scale=1.0, size=(N, 2))
    optimum = np.array([0.0, 0.0], dtype=float)
    shift = np.array(shift_per_step, dtype=float)

    # We'll keep a history if we need global or diag data
    phenotypes_history = []
    optimum_history = []
    fitness_history = []  # store current generation's fitness
    offspring_counts_history = []  # store how many offspring each individual got

    # This will hold the final list of created frame filenames (for main GIF)
    frame_filenames = []

    ##########
    # 1) RUN THE SIMULATION (collect data)
    ##########
    if not global_view:
        # ----- Normal (Per-Generation) Approach -----
        for t in range(T):
            fitnesses = fitness_function(phenotypes, optimum, alpha)

            # Reproduction (track offspring distribution for diagnostics)
            new_gen, offspring_counts = select_and_reproduce(phenotypes, fitnesses, N)
            
            # Store data for diagnostic plotting
            phenotypes_history.append(phenotypes.copy())
            optimum_history.append(optimum.copy())
            fitness_history.append(fitnesses.copy())
            offspring_counts_history.append(offspring_counts)

            # Immediately plot & save population frame
            fig, ax = plt.subplots(figsize=(5, 5))
            sc = ax.scatter(
                phenotypes[:, 0],
                phenotypes[:, 1],
                c=fitnesses,
                cmap='viridis',
                s=20
            )
            plt.colorbar(sc, ax=ax, label='Fitness')
            ax.scatter(optimum[0], optimum[1], c='red', marker='*', s=200, label='Optimum')
            ax.set_title(f"Time = {t}")
            ax.set_xlabel("Phenotype X")
            ax.set_ylabel("Phenotype Y")
            ax.legend()

            # Dynamic axis range for this time step
            min_x = min(phenotypes[:, 0].min(), optimum[0])
            max_x = max(phenotypes[:, 0].max(), optimum[0])
            min_y = min(phenotypes[:, 1].min(), optimum[1])
            max_y = max(phenotypes[:, 1].max(), optimum[1])
            margin_x = (max_x - min_x) * 0.2
            margin_y = (max_y - min_y) * 0.2
            if margin_x == 0:
                margin_x = 0.5
            if margin_y == 0:
                margin_y = 0.5
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
            ax.set_aspect('equal', 'box')

            frame_filename = os.path.join(frames_dir, f"frame_{t:03d}_{timestamp_str}.png")
            plt.savefig(frame_filename, dpi=120)
            plt.close(fig)
            frame_filenames.append(frame_filename)

            # Mutation
            phenotypes = mutate(new_gen, mutation_std)
            # Shift the optimum
            optimum += shift

            # Update progress bar if available
            if progress_bar is not None:
                progress_value = int(((t + 1) / T) * 100)
                progress_bar.progress(progress_value)

    else:
        # ===== Global View Approach =====
        # Step 1: run entire GA, store data (no plots yet)
        for t in range(T):
            fitnesses = fitness_function(phenotypes, optimum, alpha)
            new_gen, offspring_counts = select_and_reproduce(phenotypes, fitnesses, N)

            phenotypes_history.append(phenotypes.copy())
            optimum_history.append(optimum.copy())
            fitness_history.append(fitnesses.copy())
            offspring_counts_history.append(offspring_counts)

            phenotypes = mutate(new_gen, mutation_std)
            optimum += shift

            # progress for simulation phase
            if progress_bar is not None:
                # We can treat this entire phase as 50% of progress, for example
                progress_value = int(((t + 1) / T) * 50)
                progress_bar.progress(progress_value)

        # Step 2: determine global bounding box from all time steps
        all_x = []
        all_y = []
        for t in range(T):
            all_x.extend(phenotypes_history[t][:, 0].tolist())
            all_y.extend(phenotypes_history[t][:, 1].tolist())
            all_x.append(optimum_history[t][0])
            all_y.append(optimum_history[t][1])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        margin_x = (max_x - min_x) * 0.2
        margin_y = (max_y - min_y) * 0.2
        if margin_x == 0:
            margin_x = 0.5
        if margin_y == 0:
            margin_y = 0.5

        # Step 3: generate each frame with the global bounding box
        for t in range(T):
            phenos_t = phenotypes_history[t]
            opt_t = optimum_history[t]
            fitnesses_t = fitness_history[t]

            fig, ax = plt.subplots(figsize=(5, 5))
            sc = ax.scatter(
                phenos_t[:, 0],
                phenos_t[:, 1],
                c=fitnesses_t,
                cmap='viridis',
                s=20
            )
            plt.colorbar(sc, ax=ax, label='Fitness')
            ax.scatter(opt_t[0], opt_t[1], c='red', marker='*', s=200, label='Optimum')
            ax.set_title(f"Time = {t}")
            ax.set_xlabel("Phenotype X")
            ax.set_ylabel("Phenotype Y")
            ax.legend()

            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
            ax.set_aspect('equal', 'box')

            frame_filename = os.path.join(frames_dir, f"frame_{t:03d}_{timestamp_str}.png")
            plt.savefig(frame_filename, dpi=120)
            plt.close(fig)
            frame_filenames.append(frame_filename)

            # progress for frame-creation phase
            if progress_bar is not None:
                # second half of progress bar
                progress_value = 50 + int(((t + 1) / T) * 50)
                progress_bar.progress(progress_value)

    ##########
    # 2) CREATE MAIN GIF
    ##########
    with imageio.get_writer(main_gif_path, mode='I', duration=0.3) as writer:
        for filename in frame_filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)

    # Clean up the frames for the main evolution
    for filename in frame_filenames:
        if os.path.exists(filename):
            os.remove(filename)

    ##########
    # 3) DIAGNOSTIC PLOTS (if requested)
    ##########
    if create_diag_plots:
        diag_frame_filenames = []

        # Precompute time-series data
        # number of winners, average offspring, stdev offspring
        num_winners_history = []
        avg_offspring_history = []
        std_offspring_history = []

        # For each generation t
        for t in range(T):
            counts = offspring_counts_history[t]  # array of length N
            winners = np.sum(counts > 0)
            num_winners_history.append(winners)
            avg_offspring_history.append(np.mean(counts))
            std_offspring_history.append(np.std(counts))

        # Now create frames for each generation t
        # We'll show the partial time-series from 0..t, plus the histogram of fitness at generation t
        for t in range(T):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f"Diagnostics at Generation {t}")

            # 1) Number of winners over time
            ax0 = axes[0]
            ax0.plot(range(t+1), num_winners_history[:t+1], color='blue', marker='o')
            ax0.set_title("Number of Winners")
            ax0.set_xlabel("Generation")
            ax0.set_ylabel("Count of Individuals w/ Offspring")
            ax0.set_xlim(0, T)
            ax0.set_ylim(0, N+1)  # can't exceed N

            # 2) Average offspring (line) +/- std dev (shaded)
            ax1 = axes[1]
            gen_range = np.arange(t+1)
            avg_vals = np.array(avg_offspring_history[:t+1])
            std_vals = np.array(std_offspring_history[:t+1])
            ax1.plot(gen_range, avg_vals, color='green', label="Mean offspring")
            ax1.fill_between(gen_range,
                             avg_vals - std_vals,
                             avg_vals + std_vals,
                             alpha=0.2,
                             color='green',
                             label="±1 std")
            ax1.set_title("Offspring Distribution")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Offspring Count")
            ax1.legend(loc="upper left")
            ax1.set_xlim(0, T)

            # 3) Fitness distribution (histogram) at generation t
            ax2 = axes[2]
            fit_t = fitness_history[t]
            ax2.hist(fit_t, bins=30, range=(0.0, 1.0), color='orange', alpha=0.7)
            ax2.set_title(f"Fitness Dist. at t={t}")
            ax2.set_xlabel("Fitness")
            ax2.set_ylabel("Count")
            ax2.set_xlim(0, 1.05)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

            diag_frame_name = os.path.join(frames_dir, f"diag_frame_{t:03d}_{timestamp_str}.png")
            plt.savefig(diag_frame_name, dpi=120)
            plt.close(fig)
            diag_frame_filenames.append(diag_frame_name)

        # Create the diagnostic GIF
        with imageio.get_writer(diag_gif_path, mode='I', duration=0.3) as writer:
            for filename in diag_frame_filenames:
                image = imageio.v2.imread(filename)
                writer.append_data(image)

        # Clean up diagnostic frames
        for filename in diag_frame_filenames:
            if os.path.exists(filename):
                os.remove(filename)

    return main_gif_path, diag_gif_path

########################
# Streamlit Components #
########################

def parse_gif_filename(filename):
    """
    Parse a filename of the form:
      ga_simulation_{N}_{T}_{alpha}_{mutation_std}_{shift_x}_{shift_y}_{timestamp}.gif
    or diag_ + that.
    Returns a dict of parsed parameters or None if parse fails.
    """
    # If it's a diagnostic GIF, remove the "diag_" prefix for parsing
    diag_prefix = "diag_"
    was_diag = filename.startswith(diag_prefix)
    if was_diag:
        filename_for_parse = filename[len(diag_prefix):]
    else:
        filename_for_parse = filename

    pattern = (
        r"^ga_simulation_"
        r"(\d+)_"               # N
        r"(\d+)_"               # T
        r"([\-\d\.]+)_"         # alpha
        r"([\-\d\.]+)_"         # mutation_std
        r"([\-\d\.]+)_"         # shift_x
        r"([\-\d\.]+)_"         # shift_y
        r"(\d{8}_\d{6})"        # timestamp
        r"\.gif$"
    )
    match = re.match(pattern, filename_for_parse)
    if match:
        try:
            parsed = {
                "filename": filename,
                "N": int(match.group(1)),
                "T": int(match.group(2)),
                "alpha": float(match.group(3)),
                "mutation_std": float(match.group(4)),
                "shift_x": float(match.group(5)),
                "shift_y": float(match.group(6)),
                "timestamp": match.group(7),
                "is_diag": was_diag
            }
            return parsed
        except ValueError:
            return None
    return None

def filter_runs(runs, N, T, alpha, mutation_std, shift_x, shift_y, is_diag=None):
    """
    Given a list of run dictionaries (parsed from filenames) and optional
    filter parameters, return the runs that match all non-None filters.
    If a filter is None, it is ignored.
    """
    filtered = []
    for run in runs:
        if (N is not None) and (run['N'] != N):
            continue
        if (T is not None) and (run['T'] != T):
            continue
        if (alpha is not None) and (abs(run['alpha'] - alpha) > 1e-9):
            continue
        if (mutation_std is not None) and (abs(run['mutation_std'] - mutation_std) > 1e-9):
            continue
        if (shift_x is not None) and (abs(run['shift_x'] - shift_x) > 1e-9):
            continue
        if (shift_y is not None) and (abs(run['shift_y'] - shift_y) > 1e-9):
            continue
        if (is_diag is not None) and (run['is_diag'] != is_diag):
            continue

        filtered.append(run)
    return filtered

def main():
    st.title("Geometric Fisher Model Simulation (Genetic Algorithm)")

    st.markdown(
        """
        This app runs a **2D Genetic Algorithm** simulation under a 
        Geometric Fisher-like fitness function, with a shifting optimum each generation.
        
        - **Global View**: uses a single bounding box for all frames (so everything is always in view).
        - **Diagnostic Plots**: produce a second GIF with time-series of:
          - Winners (individuals that produced offspring)  
          - Average offspring \(\pm\) std dev  
          - Fitness distribution histogram
        """
    )

    st.sidebar.header("Simulation Parameters")

    # Inputs for the simulation
    N = st.sidebar.number_input("Population size (N)", min_value=1, value=500, step=1)
    T = st.sidebar.number_input("Number of time steps (T)", min_value=1, value=50, step=1)
    alpha = st.sidebar.number_input("Selection strength (alpha)", min_value=0.0, value=1.0, step=0.1)
    mutation_std = st.sidebar.number_input("Mutation std", min_value=0.0, value=0.1, step=0.01)
    shift_x = st.sidebar.number_input("Shift in X per step", value=0.05, step=0.01)
    shift_y = st.sidebar.number_input("Shift in Y per step", value=0.00, step=0.01)

    # Checkboxes for global view and diagnostic plots
    global_view = st.sidebar.checkbox("Global View (entire simulation bounding box)")
    create_diag_plots = st.sidebar.checkbox("Produce Diagnostic Plots")

    # A "Run Simulation" button
    if st.sidebar.button("Run Simulation"):
        # Create a progress bar
        progress_bar = st.progress(0)
        with st.spinner("Running simulation... (please wait)"):
            main_gif_path, diag_gif_path = run_simulation(
                N=N,
                T=T,
                alpha=alpha,
                mutation_std=mutation_std,
                shift_per_step=(shift_x, shift_y),
                frames_dir="frames",
                global_view=global_view,
                create_diag_plots=create_diag_plots,
                progress_bar=progress_bar
            )
        st.success(f"Simulation complete! Main GIF: {main_gif_path}")
        if diag_gif_path:
            st.success(f"Diagnostic GIF: {diag_gif_path}")

    st.markdown("---")
    st.header("View Existing Runs")

    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Gather all .gif files in the frames_dir
    gif_files = [f for f in os.listdir(frames_dir) if f.endswith(".gif")]
    runs_info = []
    for f in gif_files:
        parsed = parse_gif_filename(f)
        if parsed is not None:
            runs_info.append(parsed)

    # Sort runs by timestamp descending
    runs_info.sort(key=lambda x: x["timestamp"], reverse=True)

    # Filtering UI
    st.subheader("Filter Runs")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    filter_col4, filter_col5, filter_col6 = st.columns(3)

    with filter_col1:
        fN = st.number_input("Filter by N", value=0, step=1, min_value=0)
        fN = None if fN == 0 else fN
    with filter_col2:
        fT = st.number_input("Filter by T", value=0, step=1, min_value=0)
        fT = None if fT == 0 else fT
    with filter_col3:
        fAlpha = st.number_input("Filter by alpha", value=0.0, step=0.1)
        fAlpha = None if abs(fAlpha) < 1e-9 else fAlpha
    with filter_col4:
        fMutation = st.number_input("Filter by mutation_std", value=0.0, step=0.01)
        fMutation = None if abs(fMutation) < 1e-9 else fMutation
    with filter_col5:
        fShiftX = st.number_input("Filter by shift_x", value=0.0, step=0.01)
        fShiftX = None if abs(fShiftX) < 1e-9 else fShiftX
    with filter_col6:
        fShiftY = st.number_input("Filter by shift_y", value=0.0, step=0.01)
        fShiftY = None if abs(fShiftY) < 1e-9 else fShiftY

    # "Include diagnostics only" or "Exclude diagnostics" or "All"
    diag_option = st.selectbox(
        "Show Only (Main) or (Diagnostic) or (Both)?",
        ["Both", "Main Only", "Diagnostic Only"]
    )
    if diag_option == "Both":
        diag_filter = None
    elif diag_option == "Main Only":
        diag_filter = False
    else:
        diag_filter = True

    # Apply filters
    filtered_runs = filter_runs(
        runs_info,
        N=fN,
        T=fT,
        alpha=fAlpha,
        mutation_std=fMutation,
        shift_x=fShiftX,
        shift_y=fShiftY,
        is_diag=diag_filter
    )

    st.write(f"**{len(filtered_runs)}** run(s) found with current filters.")

    # Convert filtered runs to a selectable list (sorted by timestamp)
    options = [f"{r['filename']}" for r in filtered_runs]
    selected_filename = st.selectbox("Select a run to view", options)

    if selected_filename:
        match_run = next((r for r in filtered_runs if r["filename"] == selected_filename), None)
        if match_run:
            st.write("### Run Parameters")
            st.write(f"- **N** = {match_run['N']}")
            st.write(f"- **T** = {match_run['T']}")
            st.write(f"- **alpha** = {match_run['alpha']}")
            st.write(f"- **mutation_std** = {match_run['mutation_std']}")
            st.write(f"- **shift** = ({match_run['shift_x']}, {match_run['shift_y']})")
            st.write(f"- **timestamp** = {match_run['timestamp']}")
            st.write(f"- **Diagnostic GIF?** = {match_run['is_diag']}")

            gif_full_path = os.path.join(frames_dir, match_run["filename"])
            if os.path.exists(gif_full_path):
                st.image(gif_full_path, caption=f"Simulation GIF: {selected_filename}", use_column_width=True)

                # Check for matching diagnostic GIF
                diag_filename = "diag_" + match_run["filename"]
                diag_full_path = os.path.join(frames_dir, diag_filename)

                # Debugging (only shows in the Streamlit app)
                # st.write(f"Checking for Diagnostic GIF: {diag_filename}")
                # st.write(f"Full path: {diag_full_path}")
                # st.write(f"Exists? {os.path.exists(diag_full_path)}")

                # If found, display the diagnostic GIF below the main one
                if os.path.exists(diag_full_path):
                    st.image(diag_full_path, caption=f"Diagnostic GIF: {diag_filename}", use_column_width=True)
 
            else:
                st.error("GIF file not found on disk.")

if __name__ == "__main__":
    main()
