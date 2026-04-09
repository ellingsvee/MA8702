from utils import load_sensor_data


def main():
    # Load the sensor data
    sensor_data = load_sensor_data()
    print(sensor_data)


if __name__ == "__main__":
    main()
