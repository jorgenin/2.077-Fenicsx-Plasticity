from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt


plt.rcParams["animation.ffmpeg_path"] = "media/ffmpeg"


def plot_animation(
    colors, timeHist0, timeHist1, timeHist2, title1, title2, ylabel1, ylabel2
):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col")
    fig.set_size_inches(7, 5)

    def animate(i):
        plt.cla()

        ax1.plot(timeHist2[0:i], timeHist1[0:i], c=colors[3], linewidth=1.5)
        ax1.grid(linestyle="--", linewidth=0.5, color="b")
        ax1.set_ylabel(ylabel1)
        from matplotlib.ticker import AutoMinorLocator

        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

        ax2.plot(timeHist2[0:i], timeHist0[0:i], c=colors[0], linewidth=1.5)
        ax2.grid(linestyle="--", linewidth=0.5, color="b")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel(ylabel2)
        from matplotlib.ticker import AutoMinorLocator

        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())

        ax1.set_xlim(0, 2.5)
        ax1.set_ylim(-0.1, 2.251)
        ax2.set_ylim(-6.5, 6.6)

    set_fps = 24 * 3
    ani = FuncAnimation(
        fig, animate, interval=1000 / set_fps, save_count=len(timeHist0)
    )

    f = r"media/electro-viscoelastic_helix_exp_actuation_curve.mp4"
    writervideo = FFMpegWriter(fps=set_fps)
    ani.save(f, writer=writervideo)
