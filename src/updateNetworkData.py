import json
import numpy as np

class ExportUpdateNetworkData:
    def __init__(self, updateNetworkInstance):
        self.updateNetworkInstance = updateNetworkInstance

    def _serialize_numpy_array(self, obj):
        """Recursively searches for numpy arrays and float32 in the object and converts them to lists or native floats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):  # Convert np.float32 to Python float
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._serialize_numpy_array(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy_array(item) for item in obj]
        else:
            return obj
        
    def prepare_net_states_data(self):
        net_states_serialized = self._serialize_numpy_array(self.updateNetworkInstance.states)
        net_labels_serialized = self._serialize_numpy_array(self.updateNetworkInstance.net_labels)
        time_serialized = self._serialize_numpy_array(self.updateNetworkInstance.time)
        
        net_states_with_labels = {
            'x': time_serialized,
        }
        
        counter = 1
        
        # Iterate over each key and its associated values
        for key, values in net_states_serialized.items():
            if any(value >= 0.3 for value in values):  # Check if any value is >= 0.3
                label = net_labels_serialized[int(key)][10:].split(',')[0].strip()
                
                y_key = f'y{counter}'
                label_key = f'label{counter}'
                
                # Add all values for this key, since at least one meets the criterion
                net_states_with_labels[y_key] = values
                net_states_with_labels[label_key] = label
                
                counter += 1  # Increment for unique keys per label
        
        return net_states_with_labels
                    
                    
                
        

    def prepare_chart_data(self):
        """Prepares data for charting in the specified structure."""
        time_serialized = self._serialize_numpy_array(self.updateNetworkInstance.time)
        real_time_serialized = self._serialize_numpy_array(round((time_serialized / 30), 1)) 
        last_estimation_serialized = self._serialize_numpy_array(self.updateNetworkInstance.last_estimation[0])
        distances_serialized = self._serialize_numpy_array(self.updateNetworkInstance.distances)
        salient_features_serialized = self._serialize_numpy_array(self.updateNetworkInstance.salientFeatures)
        

        # Define your charts here
        chart_data = {
            'charts': [
                # Network States
                {
                    'name': 'Features',
                    'x_label': 'Frames',
                    'y_label': 'Network States',
                    'data': self.prepare_net_states_data()
                },
                
                # Euclidean Distances
                {
                    'name': 'Euclidean Distance - Layer 0',
                    'x_label': 'Frames',
                    'y_label': 'Layer 0',
                    'data': {
                        'x': time_serialized,
                        'y1': distances_serialized[0],
                        'y2': distances_serialized["T0"],
                    }
                },
                {
                    'name': 'Euclidean Distance - Layer 1',
                    'x_label': 'Frames',
                    'y_label': 'Layer 1',
                    'data': {
                        'x': time_serialized,
                        'y1': distances_serialized[1],
                        'y2': distances_serialized["T1"],
                    }
                },
                {
                    'name': 'Euclidean Distance - Layer 2',
                    'x_label': 'Frames',
                    'y_label': 'Layer 2',
                    'data': {
                        'x': time_serialized,
                        'y1': distances_serialized[2],
                        'y2': distances_serialized["T2"],
                    }
                },
                {
                    'name': 'Euclidean Distance - Layer 3',
                    'x_label': 'Frames',
                    'y_label': 'Layer 3',
                    'data': {
                        'x': time_serialized,
                        'y1': distances_serialized[3],
                        'y2': distances_serialized["T3"],
                    }
                },
                
                # Salient Features
                {
                    'name': 'Salient Features - Layer 0',
                    'x_label': 'Frames',
                    'y_label': 'Layer 0',
                    'data': {
                        'x': time_serialized,
                        'y': salient_features_serialized[0],
                    }
                },
                {
                    'name': 'Salient Features - Layer 1',
                    'x_label': 'Frames',
                    'y_label': 'Layer 1',
                    'data': {
                        'x': time_serialized,
                        'y': salient_features_serialized[1],
                    }
                },
                {
                    'name': 'Salient Features - Layer 2',
                    'x_label': 'Frames',
                    'y_label': 'Layer 2',
                    'data': {
                        'x': time_serialized,
                        'y': salient_features_serialized[2],
                    }
                },
                {
                    'name': 'Salient Features - Layer 3',
                    'x_label': 'Frames',
                    'y_label': 'Layer 3',
                    'data': {
                        'x': time_serialized,
                        'y': salient_features_serialized[3],
                    }
                },
                # Real Time vs Estimated Time
                {
                    'name': 'Real Time vs Estimated Time',
                    'x_label': 'Real Time',
                    'y_label': 'Estimated Time',
                    'data': {
                        'real': real_time_serialized,
                        'est': last_estimation_serialized,
                    }
                }        
            ]
        }
        return chart_data

    def to_json(self):
        """Converts the updateNetwork instance data to a JSON string suitable for charting, using the prepared chart data."""
        chart_data = self.prepare_chart_data()
        return json.dumps(chart_data, indent=4)



