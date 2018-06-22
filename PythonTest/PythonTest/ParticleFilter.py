
from math import *
import random

landmarks = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]]
world_size = 100.0


class robot:


    def __init__(self):

        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.forward_noise = 0.0;
        self.turn_noise = 0.0;
        self.sense_noise = 0.0;


    def set(self, new_x, new_y, new_orientation):

        if new_x < 0 or new_x >= world_size:
            raise (ValueError, 'X coordinate out of bound')

        if new_y < 0 or new_y >= world_size:
            raise (ValueError, 'Y coordinate out of bound')

        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise (ValueError, 'Orientation must be in [0..2pi]')

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)


    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):

        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise = float(new_t_noise);
        self.sense_noise = float(new_s_noise);


    def sense(self):

        Z = []

        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            Z.append(dist)

        return Z


    def move(self, turn, forward):

        if forward < 0:
            raise (ValueError, 'Robot cant move backwards')

        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi

        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        x %= world_size # cyclic truncate
        y %= world_size

        # set particle
        res = robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)

        return res


    def Gaussian(self, mu, sigma, x):

        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))


    def measurement_prob(self, measurement):

        # calculates how likely a measurement should be
        prob = 1.0;

        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)

        prob *= self.Gaussian(dist, self.sense_noise, measurement[i])

        return prob


    def __repr__(self):

        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))


def eval(r, p):

    sum = 0.0;

    for i in range(len(p)): # calculate mean error
        dx = (p[i].x - r.x + (world_size/2.0)) % world_size - (world_size/2.0)
        dy = (p[i].y - r.y + (world_size/2.0)) % world_size - (world_size/2.0)
        err = sqrt(dx * dx + dy * dy)
        sum += err

    return sum / float(len(p))

# --------

N = 1000  # number of particles
T = 10    # number of turns (moving steps)

myrobot = robot()  # initializles robot object attributes x, y, orientation, forward_noise, turn_noise and sense_noise

p = []

for i in range(N):
    r = robot()
    #r.set_noise(0.01,0.01,1.0)
    #r.set_noise(0.05,0.05,1.0)
    r.set_noise(0.05, 0.05, 5.0)  # Sebastian's provided noise. Set forward_noise, turn_noise and sense_noise, in this order
    p.append(r)  # insert into particles vector robot objects with the attributes x, y, orientation, forward_noise, turn_noise and sense_noise

# now we have 1000 particles of random coordinates within world size and with set standard noise
    
for t in range(T):  # for all 10 moves do..
    myrobot= myrobot.move(0.1, 5.0)  # move robot with turn of 0.1 and forward 5.0 and set particle with calculated x, y and orientation
    Z = myrobot.sense()  # calculate distances to 4 landmarks
    
    p2 = []  # new particle set
    
    for i in range(N):
        p2.append(p[i].move(0.1, 5.0))  # initialize new particle set (1000 particles) with actual move for all existing particles

    p = p2  # take particle set (calculated x, y and orientation) with actual move
    w = []  # weight set

    for i in range(N):
        w.append(p[i].measurement_prob(Z))  # for all of existing 1000 particles calculate how likely a measurement should be with gaussian distribution of parameters distance to landmark (mu), sense noise (sigma) and measurement (x)

    # now we have for all of the particles according weights (from gaussian distribution)

    p3 = []  # new particle set

    index = int(random.random() * N)  # set random index (0..999)
    beta = 0.0
    mw = max(w)  # get maximum weight

    for i in range(N):
        beta += random.random() * 2.0 * mw  # beta is the cumulated sum over all particles (1000) as a random double maximum weight

        while beta > w[index]:       # while beta is greater than a concrete weight do
            beta -= w[index]         # substract the concrete weight from beta and
            index = (index + 1) % N  # modulo operator to ensure index element of [1..1000]

        # now we have the first p[index] where beta <= w[index] (where cumulated sum of random double maximum weight is smaller equal the according weight on the same place
        # Wir haben dies N-mal getan und erhalten N Partikel, und wir können sehen, dass die Partikel schrittweise sortiert werden im Verhältnis zu ihrem Umfang auf dem Kreis.
        # Die Partikel sind nun der Reihe nach gruppiert sortiert infolge der 3 Dimensionen (x, y, orientation) pro Bewegung (move) bzw. Messung (sense)
        p3.append(p[index])

    p = p3

    print (eval(myrobot, p))


if eval(myrobot, p) > 15.0:  # if system diverges...

    for i in range(N):
        print ('#', i, p[i])

    print ('R', myrobot)


# OUTPUT (example):
# =======
# # 702 [x=79.340 y=49.112 orient=2.9222]
# # 703 [x=41.275 y=16.450 orient=2.3117]
# # 704 [x=42.654 y=2.9678 orient=3.1579]
# # 705 [x=79.509 y=49.101 orient=2.8587]
# # 706 [x=78.989 y=47.970 orient=3.0415]
# # 707 [x=52.745 y=19.181 orient=2.1171]
# # 708 [x=43.233 y=6.5857 orient=2.9674]
# # 709 [x=43.022 y=17.719 orient=2.1920]
# # 710 [x=78.513 y=46.424 orient=3.0953]
# # 711 [x=43.988 y=7.2325 orient=3.0385]
# # 712 [x=42.990 y=3.8789 orient=3.2164]
# # 713 [x=43.940 y=7.7580 orient=2.9459]
# # 714 [x=43.708 y=6.6087 orient=3.0532]
# # 715 [x=78.588 y=47.594 orient=2.9697]
# # 716 [x=79.246 y=48.494 orient=2.9189]
# # 717 [x=43.266 y=6.2742 orient=3.0129]
# # 718 [x=79.115 y=49.146 orient=2.9539]
# # 719 [x=79.591 y=50.154 orient=2.8559]
# # 720 [x=79.528 y=50.255 orient=2.8401]
# # 721 [x=79.528 y=50.255 orient=2.8401]
# # 722 [x=79.528 y=50.255 orient=2.8401]
# # 723 [x=42.881 y=3.7111 orient=3.1513]
# # 724 [x=53.124 y=19.440 orient=2.0683]
# # 725 [x=79.093 y=49.336 orient=2.8735]
# # 726 [x=78.949 y=48.884 orient=2.9681]
# # 727 [x=78.949 y=48.884 orient=2.9681]
# # 728 [x=78.981 y=48.933 orient=2.9573]
# # 729 [x=44.092 y=18.330 orient=2.1455]
# # 730 [x=42.091 y=16.884 orient=2.3657]
# # 731 [x=42.091 y=16.884 orient=2.3657]
# # 732 [x=78.388 y=46.702 orient=3.0507]
# # 733 [x=43.963 y=5.7420 orient=3.1318]
# # 734 [x=42.998 y=6.4675 orient=2.9702]
# # 735 [x=78.290 y=47.714 orient=2.9590]
# # 736 [x=44.285 y=7.6906 orient=3.0195]
# # 737 [x=44.097 y=6.7321 orient=3.1548]
# # 738 [x=44.274 y=7.2863 orient=3.0427]
# # 739 [x=78.779 y=48.511 orient=2.8909]
# # 740 [x=78.751 y=47.469 orient=3.0975]
# # 741 [x=78.751 y=47.469 orient=3.0975]
# # 742 [x=44.424 y=7.9795 orient=2.9458]
# # 743 [x=50.091 y=18.630 orient=2.1339]
# # 744 [x=50.091 y=18.630 orient=2.1339]
# # 745 [x=78.992 y=48.395 orient=3.0303]
# # 746 [x=42.401 y=2.3869 orient=3.1965]
# # 747 [x=43.188 y=3.4405 orient=3.3087]
# # 748 [x=43.635 y=6.8075 orient=2.9775]
# # 749 [x=77.873 y=45.731 orient=3.0697]
# # 750 [x=79.294 y=48.906 orient=2.9499]
# # 751 [x=78.482 y=47.488 orient=2.9704]
# # 752 [x=43.751 y=6.8057 orient=3.0325]
# # 753 [x=43.692 y=6.3446 orient=3.0271]
# # 754 [x=42.555 y=2.0119 orient=3.2698]
# # 755 [x=42.555 y=2.0119 orient=3.2698]
# # 756 [x=79.307 y=49.461 orient=2.9335]
# # 757 [x=79.307 y=49.461 orient=2.9335]
# # 758 [x=77.522 y=43.775 orient=3.1663]
# # 759 [x=43.267 y=5.2565 orient=3.0807]
# # 760 [x=44.661 y=8.8142 orient=2.9473]
# # 761 [x=44.661 y=8.8142 orient=2.9473]
# # 762 [x=79.102 y=47.905 orient=2.9437]
# # 763 [x=79.102 y=47.905 orient=2.9437]
# # 764 [x=79.102 y=47.905 orient=2.9437]
# # 765 [x=53.241 y=19.354 orient=2.0280]
# # 766 [x=43.871 y=6.9076 orient=3.0916]
# # 767 [x=51.734 y=19.073 orient=2.0616]
# # 768 [x=79.180 y=49.277 orient=2.9088]
# # 769 [x=79.307 y=49.264 orient=2.9113]
# # 770 [x=41.365 y=16.898 orient=2.2617]
# # 771 [x=42.644 y=3.4411 orient=3.1187]
# # 772 [x=43.782 y=4.9429 orient=3.1876]
# # 773 [x=42.605 y=4.5576 orient=3.0655]
# # 774 [x=42.555 y=2.3978 orient=3.2399]
# # 775 [x=43.748 y=6.0969 orient=3.0976]
# # 776 [x=43.748 y=6.0969 orient=3.0976]
# # 777 [x=78.896 y=47.671 orient=3.0535]
# # 778 [x=43.044 y=3.2102 orient=3.2370]
# # 779 [x=78.662 y=46.850 orient=3.0750]
# # 780 [x=44.004 y=18.234 orient=2.0855]
# # 781 [x=78.159 y=46.806 orient=2.9548]
# # 782 [x=78.159 y=46.806 orient=2.9548]
# # 783 [x=42.738 y=3.5239 orient=3.1383]
# # 784 [x=53.285 y=19.599 orient=2.0819]
# # 785 [x=78.552 y=46.976 orient=3.0202]
# # 786 [x=78.552 y=46.976 orient=3.0202]
# # 787 [x=79.372 y=49.566 orient=2.8808]
# # 788 [x=79.372 y=49.566 orient=2.8808]
# # 789 [x=78.793 y=49.017 orient=2.8523]
# # 790 [x=43.065 y=5.0614 orient=3.0742]
# # 791 [x=43.065 y=5.0614 orient=3.0742]
# # 792 [x=42.593 y=4.8252 orient=3.0146]
# # 793 [x=42.569 y=4.9734 orient=2.9862]
# # 794 [x=44.078 y=7.5802 orient=2.8888]
# # 795 [x=79.478 y=49.673 orient=2.9083]
# # 796 [x=79.411 y=50.095 orient=2.8307]
# # 797 [x=79.353 y=49.296 orient=2.9879]
# # 798 [x=79.353 y=49.296 orient=2.9879]
# # 799 [x=43.341 y=6.4754 orient=3.0165]
# # 800 [x=79.635 y=50.438 orient=2.8035]
# # 801 [x=42.761 y=5.2734 orient=3.0321]
# # 802 [x=43.731 y=6.1464 orient=3.0413]
# # 803 [x=43.597 y=18.606 orient=2.0215]
# # 804 [x=42.444 y=2.7964 orient=3.2484]
# # 805 [x=78.813 y=47.237 orient=3.1090]
# # 806 [x=78.253 y=46.548 orient=3.0825]
# # 807 [x=78.383 y=47.576 orient=3.0082]
# # 808 [x=44.517 y=9.5843 orient=2.7975]
# # 809 [x=44.517 y=9.5843 orient=2.7975]
# # 810 [x=43.914 y=5.3946 orient=3.2360]
# # 811 [x=51.556 y=19.071 orient=2.1188]
# # 812 [x=52.290 y=19.506 orient=2.1173]
# # 813 [x=42.640 y=4.0763 orient=3.1077]
# # 814 [x=43.116 y=5.7113 orient=3.0233]
# # 815 [x=44.252 y=8.1583 orient=2.9507]
# # 816 [x=43.290 y=6.6768 orient=2.9803]
# # 817 [x=78.894 y=48.159 orient=3.0065]
# # 818 [x=79.057 y=48.576 orient=2.9190]
# # 819 [x=79.057 y=48.576 orient=2.9190]
# # 820 [x=78.790 y=47.425 orient=3.0343]
# # 821 [x=79.207 y=48.830 orient=2.9372]
# # 822 [x=43.144 y=18.057 orient=2.1441]
# # 823 [x=78.894 y=48.391 orient=2.9437]
# # 824 [x=43.163 y=3.4745 orient=3.1947]
# # 825 [x=43.757 y=4.9213 orient=3.1912]
# # 826 [x=79.128 y=48.741 orient=2.9391]
# # 827 [x=78.822 y=47.136 orient=3.1144]
# # 828 [x=43.347 y=2.3403 orient=3.3478]
# # 829 [x=45.244 y=10.486 orient=2.8490]
# # 830 [x=43.720 y=6.4067 orient=3.0156]
# # 831 [x=44.290 y=7.4248 orient=2.9946]
# # 832 [x=44.135 y=7.6399 orient=2.9677]
# # 833 [x=52.906 y=19.719 orient=2.0210]
# # 834 [x=80.466 y=51.375 orient=2.7935]
# # 835 [x=80.521 y=51.529 orient=2.7606]
# # 836 [x=43.907 y=7.1853 orient=2.9765]
# # 837 [x=43.864 y=7.0730 orient=3.0000]
# # 838 [x=44.184 y=8.2963 orient=2.9344]
# # 839 [x=49.370 y=17.921 orient=2.2713]
# # 840 [x=79.415 y=49.646 orient=2.9013]
# # 841 [x=79.415 y=49.646 orient=2.9013]
# # 842 [x=42.109 y=17.408 orient=2.2157]
# # 843 [x=79.120 y=49.402 orient=2.8314]
# # 844 [x=45.305 y=18.931 orient=2.0810]
# # 845 [x=43.407 y=5.2652 orient=3.1129]
# # 846 [x=43.981 y=7.1882 orient=3.0398]
# # 847 [x=78.039 y=46.891 orient=3.0074]
# # 848 [x=79.651 y=49.982 orient=2.8555]
# # 849 [x=79.625 y=49.718 orient=2.9082]
# # 850 [x=42.941 y=4.3936 orient=3.0952]
# # 851 [x=42.704 y=4.9353 orient=3.0367]
# # 852 [x=43.667 y=5.0432 orient=3.1225]
# # 853 [x=79.013 y=48.215 orient=3.0235]
# # 854 [x=79.218 y=49.088 orient=2.9321]
# # 855 [x=43.336 y=4.2593 orient=3.1722]
# # 856 [x=79.271 y=48.536 orient=2.9794]
# # 857 [x=79.362 y=48.529 orient=2.9777]
# # 858 [x=78.919 y=48.202 orient=2.9983]
# # 859 [x=43.168 y=6.4865 orient=2.9891]
# # 860 [x=42.992 y=17.818 orient=2.1854]
# # 861 [x=43.462 y=5.5147 orient=3.0027]
# # 862 [x=77.852 y=45.000 orient=3.1204]
# # 863 [x=43.920 y=7.1103 orient=3.0474]
# # 864 [x=43.920 y=7.1103 orient=3.0474]
# # 865 [x=48.739 y=17.493 orient=2.3099]
# # 866 [x=43.606 y=6.5105 orient=3.0744]
# # 867 [x=79.075 y=48.377 orient=2.9492]
# # 868 [x=42.439 y=2.8062 orient=3.1669]
# # 869 [x=42.409 y=3.2761 orient=3.1126]
# # 870 [x=42.403 y=2.3651 orient=3.2909]
# # 871 [x=78.733 y=48.086 orient=2.9896]
# # 872 [x=79.087 y=49.435 orient=2.8986]
# # 873 [x=79.647 y=50.279 orient=2.8284]
# # 874 [x=79.647 y=50.279 orient=2.8284]
# # 875 [x=79.462 y=49.745 orient=2.7612]
# # 876 [x=43.003 y=3.8205 orient=3.1295]
# # 877 [x=78.691 y=48.481 orient=2.9065]
# # 878 [x=42.658 y=1.7136 orient=3.2927]
# # 879 [x=44.197 y=18.329 orient=2.1280]
# # 880 [x=44.197 y=18.329 orient=2.1280]
# # 881 [x=42.216 y=17.144 orient=2.3115]
# # 882 [x=43.613 y=6.0851 orient=3.0937]
# # 883 [x=78.191 y=47.873 orient=2.9319]
# # 884 [x=44.260 y=7.6120 orient=3.0357]
# # 885 [x=44.257 y=7.2294 orient=3.0545]
# # 886 [x=44.257 y=7.2294 orient=3.0545]
# # 887 [x=42.583 y=3.5155 orient=3.1154]
# # 888 [x=43.422 y=5.7513 orient=3.0829]
# # 889 [x=44.241 y=7.5578 orient=3.0344]
# # 890 [x=50.203 y=18.596 orient=2.1183]
# # 891 [x=78.975 y=48.518 orient=3.0063]
# # 892 [x=78.975 y=48.518 orient=3.0063]
# # 893 [x=78.996 y=48.254 orient=3.0583]
# # 894 [x=78.652 y=48.414 orient=2.9062]
# # 895 [x=43.361 y=3.8187 orient=3.2389]
# # 896 [x=43.394 y=5.2685 orient=3.0920]
# # 897 [x=43.989 y=7.3810 orient=3.0165]
# # 898 [x=42.245 y=1.0057 orient=3.1666]
# # 899 [x=78.263 y=47.143 orient=3.0439]
# # 900 [x=79.221 y=48.649 orient=3.0028]
# # 901 [x=79.221 y=48.649 orient=3.0028]
# # 902 [x=43.327 y=18.405 orient=2.1600]
# # 903 [x=43.327 y=18.405 orient=2.1600]
# # 904 [x=78.162 y=46.788 orient=2.9328]
# # 905 [x=43.537 y=4.2138 orient=3.2034]
# # 906 [x=79.416 y=49.326 orient=2.8892]
# # 907 [x=78.633 y=48.960 orient=2.8426]
# # 908 [x=78.633 y=48.960 orient=2.8426]
# # 909 [x=78.638 y=48.727 orient=2.8870]
# # 910 [x=43.187 y=6.6264 orient=3.0255]
# # 911 [x=42.470 y=3.0448 orient=3.1774]
# # 912 [x=50.892 y=18.188 orient=2.1758]
# # 913 [x=44.655 y=9.1226 orient=2.8874]
# # 914 [x=78.919 y=47.338 orient=3.0641]
# # 915 [x=78.919 y=47.338 orient=3.0641]
# # 916 [x=44.008 y=18.172 orient=2.0784]
# # 917 [x=43.665 y=6.5885 orient=3.0681]
# # 918 [x=43.835 y=7.3960 orient=2.9947]
# # 919 [x=51.152 y=18.696 orient=2.1993]
# # 920 [x=79.036 y=49.161 orient=2.9376]
# # 921 [x=79.179 y=48.898 orient=2.9885]
# # 922 [x=78.780 y=48.566 orient=2.9199]
# # 923 [x=43.727 y=6.3661 orient=3.0443]
# # 924 [x=44.466 y=8.5960 orient=2.9331]
# # 925 [x=44.466 y=8.5960 orient=2.9331]
# # 926 [x=78.904 y=47.795 orient=3.0287]
# # 927 [x=42.941 y=3.4588 orient=3.1858]
# # 928 [x=78.703 y=47.098 orient=3.0247]
# # 929 [x=78.703 y=47.098 orient=3.0247]
# # 930 [x=42.713 y=3.5653 orient=3.2058]
# # 931 [x=42.713 y=3.5653 orient=3.2058]
# # 932 [x=42.914 y=3.5272 orient=3.2526]
# # 933 [x=43.424 y=17.937 orient=2.1956]
# # 934 [x=43.795 y=18.212 orient=2.1236]
# # 935 [x=43.915 y=6.7518 orient=3.0364]
# # 936 [x=79.682 y=49.864 orient=2.8872]
# # 937 [x=79.611 y=50.150 orient=2.8518]
# # 938 [x=79.578 y=50.094 orient=2.8643]
# # 939 [x=79.232 y=49.261 orient=2.9465]
# # 940 [x=79.232 y=49.261 orient=2.9465]
# # 941 [x=43.403 y=6.3252 orient=2.9224]
# # 942 [x=80.198 y=51.510 orient=2.7756]
# # 943 [x=43.055 y=3.9760 orient=3.1769]
# # 944 [x=43.977 y=7.1480 orient=2.9773]
# # 945 [x=78.064 y=46.804 orient=3.0384]
# # 946 [x=77.906 y=45.911 orient=3.0696]
# # 947 [x=79.430 y=49.918 orient=2.8631]
# # 948 [x=79.430 y=49.918 orient=2.8631]
# # 949 [x=43.244 y=6.6440 orient=2.9530]
# # 950 [x=79.984 y=51.099 orient=2.6545]
# # 951 [x=42.576 y=1.7899 orient=3.2569]
# # 952 [x=43.515 y=5.0249 orient=3.1057]
# # 953 [x=52.017 y=19.524 orient=2.1096]
# # 954 [x=78.763 y=47.426 orient=3.0716]
# # 955 [x=78.407 y=47.866 orient=2.9498]
# # 956 [x=78.407 y=47.866 orient=2.9498]
# # 957 [x=44.301 y=8.8482 orient=2.9509]
# # 958 [x=44.301 y=8.8482 orient=2.9509]
# # 959 [x=44.328 y=9.0387 orient=2.9127]
# # 960 [x=44.039 y=7.3810 orient=3.0878]
# # 961 [x=44.123 y=7.1868 orient=3.1394]
# # 962 [x=52.067 y=19.336 orient=2.0033]
# # 963 [x=43.609 y=5.0958 orient=3.1522]
# # 964 [x=43.689 y=4.8027 orient=3.2112]
# # 965 [x=43.689 y=4.8027 orient=3.2112]
# # 966 [x=43.677 y=6.4487 orient=2.9771]
# # 967 [x=42.608 y=3.4457 orient=3.1824]
# # 968 [x=42.608 y=3.4457 orient=3.1824]
# # 969 [x=43.030 y=5.0604 orient=3.1557]
# # 970 [x=43.797 y=18.208 orient=2.1457]
# # 971 [x=43.782 y=17.986 orient=2.1371]
# # 972 [x=43.722 y=6.5553 orient=3.0366]
# # 973 [x=43.722 y=6.5553 orient=3.0366]
# # 974 [x=79.046 y=48.876 orient=2.9270]
# # 975 [x=79.046 y=48.876 orient=2.9270]
# # 976 [x=78.879 y=48.442 orient=2.9520]
# # 977 [x=78.966 y=48.642 orient=2.8862]
# # 978 [x=78.966 y=48.642 orient=2.8862]
# # 979 [x=79.045 y=48.425 orient=3.0219]
# # 980 [x=79.421 y=48.786 orient=2.9454]
# # 981 [x=79.421 y=48.786 orient=2.9454]
# # 982 [x=43.470 y=18.177 orient=2.0762]
# # 983 [x=43.470 y=18.177 orient=2.0762]
# # 984 [x=43.695 y=6.9068 orient=2.9405]
# # 985 [x=43.505 y=5.8177 orient=3.0579]
# # 986 [x=42.922 y=4.9652 orient=3.1303]
# # 987 [x=42.922 y=4.9652 orient=3.1303]
# # 988 [x=43.762 y=5.8803 orient=3.0927]
# # 989 [x=43.767 y=5.8734 orient=3.0941]
# # 990 [x=78.870 y=47.561 orient=3.0289]
# # 991 [x=78.870 y=47.561 orient=3.0289]
# # 992 [x=45.170 y=10.180 orient=2.9124]
# # 993 [x=42.600 y=3.5558 orient=3.1322]
# # 994 [x=39.009 y=16.322 orient=2.3073]
# # 995 [x=43.589 y=6.0003 orient=3.0709]
# # 996 [x=52.495 y=19.563 orient=2.1088]
# # 997 [x=80.032 y=50.667 orient=2.8791]
# # 998 [x=42.114 y=0.3759 orient=3.3252]
# # 999 [x=43.679 y=6.7522 orient=3.0448]
# R [x=80.696 y=52.296 orient=2.7891]
# Press any key to continue . . .