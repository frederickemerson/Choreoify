import cv2
from ultralytics import YOLO
import torch
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import cv2
import librosa
from scipy import signal
import tempfile
import subprocess
import os
from scipy.io import wavfile
from pathlib import Path

def dtw_path(distance_matrix):
    """
    Compute the DTW path through a distance matrix
    Returns a list of (i, j) tuples representing the path
    """
    n, m = distance_matrix.shape
    # Initialize cost matrix
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0
    
    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = distance_matrix[i-1, j-1] + min(
                cost[i-1, j],    # insertion
                cost[i, j-1],    # deletion
                cost[i-1, j-1]   # match
            )
    
    # Traceback
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        min_cost = min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        if min_cost == cost[i-1, j-1]:
            i, j = i-1, j-1
        elif min_cost == cost[i-1, j]:
            i -= 1
        else:
            j -= 1
    
    return path[::-1] 

def test_detection(video_path='./ref-2.mp4', max_frames=3000, save_interval=30):
    # Initialize YOLO model
    model = YOLO('yolo11m-pose.pt')
    model.conf = 0.25
    model.imgsz = (1080, 774)
    
    # Create debug window
    cv2.namedWindow('Debug View', cv2.WINDOW_NORMAL)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")
    
    # Initialize video writer
    output_path = 'ref_2_detection_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Define colors for different people
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    # Define keypoint names for better readability
    keypoint_names = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
    
    # Define skeleton connections
    skeleton = [
        (5, 7),   # Left shoulder to left elbow
        (7, 9),   # Left elbow to left wrist
        (6, 8),   # Right shoulder to right elbow
        (8, 10),  # Right elbow to right wrist
        (5, 6),   # Left shoulder to right shoulder
        (5, 11),  # Left shoulder to left hip
        (6, 12),  # Right shoulder to right hip
        (11, 12), # Left hip to right hip
        (11, 13), # Left hip to left knee
        (13, 15), # Left knee to left ankle
        (12, 14), # Right hip to right knee
        (14, 16)  # Right knee to right ankle
    ]
    
    # Initialize storage for pose data and person tracking
    pose_data = defaultdict(lambda: defaultdict(list))
    last_known_positions = {}  # Store last known positions of each person
    person_trackers = {}  # Store persistent IDs
    next_person_id = 1
    frame_count = 0
    
    def get_person_center(keypoints):
        """Calculate the center point of a person based on their keypoints"""
        valid_points = []
        for kp in keypoints:
            if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                valid_points.append((kp[0], kp[1]))
        if not valid_points:
            return None
        return (
            sum(p[0] for p in valid_points) / len(valid_points),
            sum(p[1] for p in valid_points) / len(valid_points)
        )
    
    def assign_person_id(current_center, last_positions, max_distance=100):
        """Assign consistent ID based on proximity to last known positions"""
        if not last_positions:
            return None
            
        # Find the closest previous person
        min_dist = float('inf')
        closest_id = None
        
        for person_id, last_pos in last_positions.items():
            dist = np.sqrt((current_center[0] - last_pos[0])**2 + 
                         (current_center[1] - last_pos[1])**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                closest_id = person_id
                
        return closest_id
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"\rProcessing frame {frame_count}", end="")
        
        # Create debug visualization
        debug_frame = frame.copy()
        
        # Run detection
        results = model(frame, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            
            # Process detections
            if result.boxes is not None and len(result.boxes) > 0:
                num_people = len(result.boxes)
                current_frame_positions = {}
                
                # First pass: calculate centers and assign/maintain IDs
                if result.keypoints is not None:
                    keypoints = result.keypoints.data
                    if torch.is_tensor(keypoints):
                        keypoints = keypoints.cpu().numpy()
                    
                    # Process each person
                    for person_idx, person_keypoints in enumerate(keypoints):
                        center = get_person_center(person_keypoints)
                        if center is None:
                            continue
                            
                        # Try to match with existing person
                        matched_id = assign_person_id(center, last_known_positions)
                        
                        # If no match found, assign new ID
                        if matched_id is None:
                            matched_id = next_person_id
                            next_person_id += 1
                        
                        # Store the ID and update position
                        person_trackers[person_idx] = matched_id
                        current_frame_positions[matched_id] = center
                
                # Second pass: draw boxes with consistent IDs
                for person_idx, box in enumerate(result.boxes):
                    if person_idx not in person_trackers:
                        continue
                        
                    person_id = person_trackers[person_idx]
                    color = colors[(person_id - 1) % len(colors)]
                    
                    # Draw bounding box
                    box_data = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box_data)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add consistent person ID
                    cv2.putText(debug_frame, f"Person {person_id}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)
                
                # Update last known positions for next frame
                last_known_positions = current_frame_positions.copy()
            
            # Process and store keypoints
            if result.keypoints is not None:
                keypoints = result.keypoints.data
                if torch.is_tensor(keypoints):
                    keypoints = keypoints.cpu().numpy()
                
                # Store pose data at intervals
                if frame_count % save_interval == 0:
                    timestamp = frame_count / fps
                    
                    for person_idx, person_keypoints in enumerate(keypoints):
                        # Convert keypoints to dictionary format
                        keypoints_dict = {
                            keypoint_names[i]: {
                                'x': float(kp[0]),
                                'y': float(kp[1]),
                                'confidence': float(kp[2]) if len(kp) > 2 else None
                            }
                            for i, kp in enumerate(person_keypoints)
                            if len(kp) >= 2
                        }
                        
                        # Store with timestamp and consistent ID
                        person_id = person_trackers.get(person_idx, person_idx + 1)
                        pose_data[f"person_{person_id}"][frame_count] = {
                            'timestamp': timestamp,
                            'keypoints': keypoints_dict
                        }
                
                # Draw keypoints and skeletons
                for person_idx, person_keypoints in enumerate(keypoints):
                    color = colors[person_idx % len(colors)]
                    
                    # Draw keypoints
                    for kp in person_keypoints:
                        if len(kp) >= 2:
                            x, y = int(kp[0]), int(kp[1])
                            if x > 0 and y > 0:
                                cv2.circle(debug_frame, (x, y), 3, color, -1)
                    
                    # Draw skeleton
                    for start_idx, end_idx in skeleton:
                        if (start_idx < len(person_keypoints) and 
                            end_idx < len(person_keypoints)):
                            start_point = person_keypoints[start_idx]
                            end_point = person_keypoints[end_idx]
                            
                            if (len(start_point) >= 2 and len(end_point) >= 2):
                                x1, y1 = int(start_point[0]), int(start_point[1])
                                x2, y2 = int(end_point[0]), int(end_point[1])
                                
                                if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0):
                                    cv2.line(debug_frame, (x1, y1), 
                                           (x2, y2), color, 2)
        
        # Add frame information
        cv2.putText(debug_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show debug view
        cv2.imshow('Debug View', debug_frame)
        
        # Write frame to output video
        out.write(debug_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Save first frame for debugging
        if frame_count == 1:
            cv2.imwrite('multi_person_debug_frame.jpg', debug_frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save pose data to JSON file
    output_filename = f'pose_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_filename, 'w') as f:
        json.dump(pose_data, f, indent=2)
    
    print(f"\n\nProcessing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Found {len(pose_data)} unique persons")
    print(f"Pose data saved to {output_filename}")
    
    # Print summary statistics
    print("\nPose Data Summary:")
    for person_id, frames in pose_data.items():
        print(f"\n{person_id}:")
        print(f"  Total frames captured: {len(frames)}")
        print(f"  First appearance: Frame {min(frames.keys())}")
        print(f"  Last appearance: Frame {max(frames.keys())}")
        
        # Calculate average positions and movement statistics
        all_x = []
        all_y = []
        joint_movements = defaultdict(list)
        
        for frame_data in frames.values():
            for joint_name, joint_data in frame_data['keypoints'].items():
                if joint_data['x'] > 0 and joint_data['y'] > 0:
                    all_x.append(joint_data['x'])
                    all_y.append(joint_data['y'])
                    
                    # Track joint-specific movements
                    joint_movements[joint_name].append((joint_data['x'], joint_data['y']))
        
        # Calculate overall statistics
        if all_x and all_y:
            avg_x = sum(all_x) / len(all_x)
            avg_y = sum(all_y) / len(all_y)
            print(f"  Average position: ({avg_x:.2f}, {avg_y:.2f})")
            
            # Calculate movement range
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            print(f"  Movement range: x={x_range:.2f}, y={y_range:.2f}")
        
        # Calculate joint-specific statistics
        print("\n  Joint-specific statistics:")
        for joint_name, positions in joint_movements.items():
            if positions:
                x_coords, y_coords = zip(*positions)
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)
                
                # Calculate total distance moved
                total_distance = 0
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    total_distance += np.sqrt(dx*dx + dy*dy)
                
                print(f"    {joint_name}:")
                print(f"      Range of motion: x={x_range:.2f}, y={y_range:.2f}")
                print(f"      Total distance moved: {total_distance:.2f}")

def calculate_pose_similarity(ref_keypoints, comp_keypoints):
    """
    Calculate similarity between two sets of keypoints
    Returns a similarity score between 0 and 1
    """
    total_distance = 0
    valid_points = 0
    
    for i in range(min(len(ref_keypoints), len(comp_keypoints))):
        ref_kp = ref_keypoints[i]
        comp_kp = comp_keypoints[i]
        
        if (len(ref_kp) >= 2 and len(comp_kp) >= 2 and 
            ref_kp[0] > 0 and ref_kp[1] > 0 and 
            comp_kp[0] > 0 and comp_kp[1] > 0):
            
            # Calculate Euclidean distance
            distance = np.sqrt(
                (ref_kp[0] - comp_kp[0])**2 + 
                (ref_kp[1] - comp_kp[1])**2
            )
            
            # Normalize by frame dimensions
            normalized_distance = distance / np.sqrt(1080**2 + 774**2)
            total_distance += normalized_distance
            valid_points += 1
    
    if valid_points == 0:
        return 0
    
    # Convert distance to similarity
    similarity = 1 - (total_distance / valid_points)
    return max(0, min(1, similarity))

def detect_movement_start(keypoints_sequence, window_size=30, threshold=0.1):
    """
    Detect when significant dance movement starts based on keypoint motion.
    
    Args:
        keypoints_sequence: List of keypoints over time
        window_size: Number of frames to analyze
        threshold: Motion threshold to consider as dance start
    
    Returns:
        start_frame: Frame index where significant movement starts
    """
    if len(keypoints_sequence) < window_size:
        return 0
        
    motion_scores = []
    
    for i in range(len(keypoints_sequence) - 1):
        curr_frame = keypoints_sequence[i]
        next_frame = keypoints_sequence[i + 1]
        
        if curr_frame is None or next_frame is None:
            motion_scores.append(0)
            continue
            
        # Calculate motion between frames using key joints
        motion = 0
        valid_points = 0
        for j in range(min(len(curr_frame), len(next_frame))):
            if (len(curr_frame[j]) >= 2 and len(next_frame[j]) >= 2 and 
                all(x > 0 for x in curr_frame[j][:2]) and 
                all(x > 0 for x in next_frame[j][:2])):
                
                dist = np.sqrt(
                    (curr_frame[j][0] - next_frame[j][0])**2 + 
                    (curr_frame[j][1] - next_frame[j][1])**2
                )
                motion += dist
                valid_points += 1
        
        if valid_points > 0:
            motion_scores.append(motion / valid_points)
        else:
            motion_scores.append(0)
    
    # Use moving average to smooth motion scores
    motion_scores = np.array(motion_scores)
    smooth_scores = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='valid')
    
    # Find where motion exceeds threshold
    start_indices = np.where(smooth_scores > threshold)[0]
    if len(start_indices) > 0:
        return start_indices[0]
    
    return 0

def extract_motion_signature(keypoints, frame_width, frame_height):
    """
    Extract normalized motion signature from keypoints
    """
    if keypoints is None or len(keypoints) == 0:
        return None
        
    signature = []
    # Use key joints for movement signature
    key_joints = [5, 6, 9, 10, 13, 14, 15, 16]  # shoulders, wrists, knees, ankles
    
    for idx in key_joints:
        if idx < len(keypoints) and len(keypoints[idx]) >= 2:
            x, y = keypoints[idx][:2]
            if x > 0 and y > 0:
                # Normalize coordinates
                x_norm = x / frame_width
                y_norm = y / frame_height
                signature.extend([x_norm, y_norm])
            else:
                signature.extend([0, 0])
        else:
            signature.extend([0, 0])
    
    return np.array(signature)

def calculate_pose_difference(ref_kp, comp_kp):
    """
    Calculate normalized difference between two poses
    Returns difference score and per-joint differences
    """
    if ref_kp is None or comp_kp is None:
        return 1.0, {}
        
    joint_differences = {}
    valid_diffs = []
    
    # Key joints to focus on (shoulders, elbows, wrists, hips, knees, ankles)
    important_joints = {
        5: "left_shoulder", 
        6: "right_shoulder",
        7: "left_elbow", 
        8: "right_elbow",
        9: "left_wrist", 
        10: "right_wrist",
        11: "left_hip", 
        12: "right_hip",
        13: "left_knee", 
        14: "right_knee",
        15: "left_ankle", 
        16: "right_ankle"
    }
    
    for idx, joint_name in important_joints.items():
        if idx < len(ref_kp) and idx < len(comp_kp):
            ref_point = ref_kp[idx]
            comp_point = comp_kp[idx]
            
            if (len(ref_point) >= 2 and len(comp_point) >= 2 and 
                all(x > 0 for x in ref_point[:2]) and 
                all(x > 0 for x in comp_point[:2])):
                
                # Calculate Euclidean distance
                diff = np.sqrt(
                    (ref_point[0] - comp_point[0])**2 + 
                    (ref_point[1] - comp_point[1])**2
                )
                
                # Normalize by diagonal of frame
                norm_diff = diff / np.sqrt(1080**2 + 774**2)
                joint_differences[joint_name] = norm_diff
                valid_diffs.append(norm_diff)
    
    if not valid_diffs:
        return 1.0, {}
    
    # Overall difference is average of joint differences
    overall_diff = np.mean(valid_diffs)
    return overall_diff, joint_differences

def calculate_joint_angles(keypoints):
    """
    Calculate angles between connected joints
    Returns a dictionary of angles for key joint connections
    
    Args:
        keypoints: numpy array of shape (N, 3) where N is number of keypoints
                  each point has (x, y, confidence)
    Returns:
        Dictionary of angles in degrees for key joint groups
    """
    def get_angle(p1, p2, p3):
        """Calculate angle between three points in degrees"""
        if len(p1) < 2 or len(p2) < 2 or len(p3) < 2:
            return None
            
        # Check if any point is invalid (0,0)
        if any(p1[:2] == [0,0]) or any(p2[:2] == [0,0]) or any(p3[:2] == [0,0]):
            return None
            
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        
        # Avoid division by zero
        if np.all(v1 == 0) or np.all(v2 == 0):
            return None
            
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Handle numerical errors
        cosine = min(1.0, max(-1.0, cosine))
        angle = np.degrees(np.arccos(cosine))
        return angle

    angles = {}
    
    # Key joint groups to analyze
    joint_groups = {
        'left_arm': (5, 7, 9),    # shoulder -> elbow -> wrist
        'right_arm': (6, 8, 10),
        'left_leg': (11, 13, 15), # hip -> knee -> ankle
        'right_leg': (12, 14, 16),
        'torso': (0, 5, 6),       # nose -> shoulders
        'hips': (11, 12, 14),     # hips -> right knee
        'shoulders': (7, 5, 6)     # left elbow -> shoulders
    }
    
    for group_name, (p1_idx, p2_idx, p3_idx) in joint_groups.items():
        if all(idx < len(keypoints) for idx in [p1_idx, p2_idx, p3_idx]):
            angle = get_angle(
                keypoints[p1_idx],
                keypoints[p2_idx],
                keypoints[p3_idx]
            )
            if angle is not None:
                angles[group_name] = angle
    
    return angles

def calculate_joint_velocities(current_keypoints, prev_keypoints, fps=30):
    """
    Calculate velocities of key joints between frames
    
    Args:
        current_keypoints: Current frame keypoints
        prev_keypoints: Previous frame keypoints
        fps: Video frame rate
    Returns:
        Dictionary of joint velocities (pixels per second)
    """
    if prev_keypoints is None:
        return {}
        
    velocities = {}
    key_joints = {
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
    
    for idx, joint_name in key_joints.items():
        if idx < len(current_keypoints) and idx < len(prev_keypoints):
            curr_pt = current_keypoints[idx]
            prev_pt = prev_keypoints[idx]
            
            if (len(curr_pt) >= 2 and len(prev_pt) >= 2 and 
                all(x > 0 for x in curr_pt[:2]) and 
                all(x > 0 for x in prev_pt[:2])):
                
                # Calculate displacement vector
                displacement = curr_pt[:2] - prev_pt[:2]
                # Convert to velocity (pixels per second)
                velocity = np.linalg.norm(displacement) * fps
                velocities[joint_name] = velocity
    
    return velocities

def compare_poses(ref_kp, comp_kp, ref_prev_kp=None, comp_prev_kp=None, fps=30):
    """
    Compare two poses using angles and velocities
    
    Args:
        ref_kp: Reference pose keypoints
        comp_kp: Comparison pose keypoints
        ref_prev_kp: Previous frame reference keypoints (for velocity)
        comp_prev_kp: Previous frame comparison keypoints (for velocity)
        fps: Video frame rate
    
    Returns:
        overall_diff: Overall difference score (0-1)
        detailed_diffs: Dictionary of specific differences
    """
    # Convert to numpy arrays if not already
    ref_kp = np.array(ref_kp)
    comp_kp = np.array(comp_kp)
    
    # Calculate angles for both poses
    ref_angles = calculate_joint_angles(ref_kp)
    comp_angles = calculate_joint_angles(comp_kp)
    
    # Calculate velocities if previous frames available
    ref_velocities = calculate_joint_velocities(ref_kp, ref_prev_kp, fps)
    comp_velocities = calculate_joint_velocities(comp_kp, comp_prev_kp, fps)
    
    detailed_diffs = {
        'angles': {},
        'velocities': {},
        'joint_positions': {}
    }
    
    # Compare angles
    angle_diffs = []
    for joint in ref_angles.keys() & comp_angles.keys():
        diff = abs(ref_angles[joint] - comp_angles[joint])
        # Normalize to 0-1 range (180 degrees max difference)
        norm_diff = min(diff / 180.0, 1.0)
        detailed_diffs['angles'][joint] = norm_diff
        angle_diffs.append(norm_diff)
    
    # Compare velocities
    velocity_diffs = []
    for joint in ref_velocities.keys() & comp_velocities.keys():
        # Normalize velocity differences relative to reference velocity
        if ref_velocities[joint] > 0:
            diff = abs(ref_velocities[joint] - comp_velocities[joint])
            norm_diff = min(diff / (ref_velocities[joint] + 1e-6), 1.0)
            detailed_diffs['velocities'][joint] = norm_diff
            velocity_diffs.append(norm_diff)
    
    # Calculate overall difference score
    # Weight angles more heavily than velocities
    angle_weight = 0.7
    velocity_weight = 0.3
    
    if angle_diffs:
        angle_score = np.mean(angle_diffs)
    else:
        angle_score = 1.0
        
    if velocity_diffs:
        velocity_score = np.mean(velocity_diffs)
    else:
        velocity_score = 1.0
    
    overall_diff = (angle_weight * angle_score + 
                   velocity_weight * velocity_score)
    
    return overall_diff, detailed_diffs

def normalize_pose_by_torso(keypoints):
    """
    Normalize pose keypoints using torso size as reference
    Makes poses comparable across different depths
    
    Args:
        keypoints: numpy array of shape (N, 3) with (x, y, confidence)
    Returns:
        normalized keypoints with same shape
    """
    # Check for valid keypoints
    if keypoints is None or len(keypoints) < 13:  # Need at least up to hips
        return None
        
    normalized = keypoints.copy()
    
    # Get shoulder and hip points
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    
    # Check if all required points are valid
    if (any(left_shoulder == 0) or any(right_shoulder == 0) or 
        any(left_hip == 0) or any(right_hip == 0)):
        return None
    
    # Calculate torso keypoints
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    
    # Calculate torso length and width
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    
    if torso_length == 0 or shoulder_width == 0:
        return None
    
    # Use torso dimensions for normalization
    scale_factor = 1.0 / torso_length
    
    # Normalize all points relative to hip center
    for i in range(len(normalized)):
        if any(normalized[i][:2] == 0):
            continue
        normalized[i][:2] = (normalized[i][:2] - hip_center) * scale_factor
    
    return normalized

def calculate_pose_orientation(keypoints):
    """
    Calculate the orientation of the pose relative to camera
    Returns angle and scaling factor based on pose orientation
    
    Args:
        keypoints: numpy array of keypoints
    Returns:
        tuple of (angle_to_camera, depth_scale)
    """
    if keypoints is None or len(keypoints) < 13:
        return None, None
        
    # Get shoulder points
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    
    if any(left_shoulder == 0) or any(right_shoulder == 0):
        return None, None
    
    # Calculate shoulder width (used for depth estimation)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    
    # Calculate angle to camera based on shoulder line
    shoulder_vector = right_shoulder - left_shoulder
    reference_vector = np.array([1, 0])  # Horizontal reference
    
    # Calculate angle
    cos_angle = np.dot(shoulder_vector, reference_vector) / (
        np.linalg.norm(shoulder_vector) * np.linalg.norm(reference_vector)
    )
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    
    # Estimate depth scale based on apparent shoulder width
    # Assuming a reference shoulder width of 100 pixels when facing camera
    reference_width = 100
    depth_scale = shoulder_width / reference_width
    
    return angle_degrees, depth_scale

def normalize_for_depth_comparison(ref_kp, comp_kp):
    """
    Normalize two poses for fair comparison across different depths
    
    Args:
        ref_kp: Reference pose keypoints
        comp_kp: Comparison pose keypoints
    Returns:
        tuple of (normalized_ref, normalized_comp, scaling_info)
    """
    # First normalize by torso
    norm_ref = normalize_pose_by_torso(ref_kp)
    norm_comp = normalize_pose_by_torso(comp_kp)
    
    if norm_ref is None or norm_comp is None:
        return None, None, None
    
    # Calculate orientations
    ref_angle, ref_depth = calculate_pose_orientation(norm_ref)
    comp_angle, comp_depth = calculate_pose_orientation(norm_comp)
    
    scaling_info = {
        'ref_angle': ref_angle,
        'comp_angle': comp_angle,
        'ref_depth': ref_depth,
        'comp_depth': comp_depth
    }
    
    return norm_ref, norm_comp, scaling_info

def adjust_joint_confidence(keypoints, angle_to_camera):
    """
    Adjust confidence scores based on pose orientation
    Reduces confidence for joints that might be occluded
    
    Args:
        keypoints: numpy array of keypoints with confidence scores
        angle_to_camera: angle in degrees
    Returns:
        keypoints with adjusted confidence scores
    """
    if angle_to_camera is None or keypoints is None:
        return keypoints
        
    adjusted = keypoints.copy()
    
    # Convert angle to 0-180 range
    angle = abs(angle_to_camera % 180)
    
    # Define joint pairs that might be occluded
    joint_pairs = [
        (5, 6),   # shoulders
        (7, 8),   # elbows
        (9, 10),  # wrists
        (11, 12), # hips
        (13, 14), # knees
        (15, 16)  # ankles
    ]
    
    for left_idx, right_idx in joint_pairs:
        if left_idx >= len(adjusted) or right_idx >= len(adjusted):
            continue
            
        # If person is turned, reduce confidence of occluded side
        if angle > 45:
            if angle < 135:  # Partially turned
                reduction = np.sin(np.radians(angle - 45)) * 0.5
                # Reduce confidence of further joints
                if angle > 90:
                    adjusted[left_idx][2] *= (1 - reduction)
                else:
                    adjusted[right_idx][2] *= (1 - reduction)
            else:  # Significantly turned
                reduction = 0.7
                # Reduce confidence of occluded joints
                if angle > 90:
                    adjusted[left_idx][2] *= (1 - reduction)
                else:
                    adjusted[right_idx][2] *= (1 - reduction)
    
    return adjusted

def find_matching_sequences(ref_keypoints, comp_keypoints, window_size=60, stride=15, similarity_threshold=0.7):
    """
    Find matching dance sequences between reference and comparison videos
    
    Args:
        ref_keypoints: List of reference pose keypoints sequences
        comp_keypoints: List of comparison pose keypoints sequences
        window_size: Size of comparison window in frames
        stride: Number of frames to slide window
        similarity_threshold: Threshold for considering sequences as matching
    Returns:
        List of (ref_start, comp_start, length) tuples indicating matching sequences
    """
    matching_sequences = []
    ref_len = len(ref_keypoints)
    comp_len = len(comp_keypoints)
    
    # Initialize previous keypoints for velocity calculation
    ref_prev = None
    comp_prev = None
    
    # Slide through both sequences
    for ref_start in range(0, ref_len - window_size, stride):
        ref_end = ref_start + window_size
        ref_sequence = ref_keypoints[ref_start:ref_end]
        
        for comp_start in range(0, comp_len - window_size, stride):
            comp_end = comp_start + window_size
            comp_sequence = comp_keypoints[comp_start:comp_end]
            
            # Track similarity scores within window
            similarities = []
            
            # Compare each frame in the window
            for i in range(window_size):
                if ref_sequence[i] is not None and comp_sequence[i] is not None:
                    # Normalize poses for depth
                    norm_ref, norm_comp, scaling_info = normalize_for_depth_comparison(
                        ref_sequence[i], comp_sequence[i]
                    )
                    
                    if norm_ref is not None and norm_comp is not None:
                        # Get previous frames for velocity calculation
                        ref_prev = ref_sequence[i-1] if i > 0 else None
                        comp_prev = comp_sequence[i-1] if i > 0 else None
                        
                        # Compare poses
                        similarity, _ = compare_poses(
                            norm_ref, norm_comp,
                            ref_prev, comp_prev
                        )
                        similarities.append(1 - similarity)  # Convert difference to similarity
            
            # Check if sequence is a good match
            if similarities and np.mean(similarities) > similarity_threshold:
                matching_sequences.append((
                    ref_start,
                    comp_start,
                    window_size,
                    np.mean(similarities)
                ))
    
    # Merge overlapping sequences
    merged_sequences = []
    matching_sequences.sort(key=lambda x: x[0])  # Sort by reference start time
    
    current_sequence = None
    for sequence in matching_sequences:
        if current_sequence is None:
            current_sequence = list(sequence)
        else:
            # Check if sequences overlap
            if (sequence[0] - (current_sequence[0] + current_sequence[2]) < window_size and
                abs((sequence[1] - current_sequence[1]) - 
                    (sequence[0] - current_sequence[0])) < window_size//2):
                # Merge sequences
                current_sequence[2] = sequence[0] + sequence[2] - current_sequence[0]
                current_sequence[3] = (current_sequence[3] + sequence[3]) / 2
            else:
                merged_sequences.append(tuple(current_sequence))
                current_sequence = list(sequence)
    
    if current_sequence is not None:
        merged_sequences.append(tuple(current_sequence))
    
    return merged_sequences

def compare_dance_videos(reference_path, comparison_path, max_frames=3000):
    """
    Compare dance videos using improved pose comparison and sequence matching
    """
    # Initialize YOLO model
    model = YOLO('yolo11m-pose.pt')
    model.conf = 0.25
    model.imgsz = (1080, 774)
    
    # Initialize video captures
    ref_cap = cv2.VideoCapture(reference_path)
    comp_cap = cv2.VideoCapture(comparison_path)
    
    if not ref_cap.isOpened() or not comp_cap.isOpened():
        print("Error: Could not open video files")
        return
    
    # Get video properties
    fps = int(ref_cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    output_path = 'dance_comparison_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))
    
    # Create debug window
    cv2.namedWindow('Comparison View', cv2.WINDOW_NORMAL)
    
    # Colors for visualization
    COLORS = {
        'good': (0, 255, 0),      # Green
        'warning': (0, 255, 255),  # Yellow
        'bad': (0, 0, 255),       # Red
        'text': (255, 255, 255)   # White
    }
    
    print("Phase 1: Collecting pose data...")
    
    # Collect pose data from both videos
    ref_keypoints = []
    comp_keypoints = []
    frame_count = 0
    
    while ref_cap.isOpened() and comp_cap.isOpened() and frame_count < max_frames:
        ref_ret, ref_frame = ref_cap.read()
        comp_ret, comp_frame = comp_cap.read()
        
        if not ref_ret or not comp_ret:
            break
            
        frame_count += 1
        print(f"\rProcessing frame {frame_count}", end="")
        
        # Process reference video
        ref_results = model(ref_frame, verbose=False)
        if len(ref_results) > 0 and ref_results[0].keypoints is not None:
            kp = ref_results[0].keypoints.data[0].cpu().numpy()
            ref_keypoints.append(kp)
        else:
            ref_keypoints.append(None)
        
        # Process comparison video
        comp_results = model(comp_frame, verbose=False)
        if len(comp_results) > 0 and comp_results[0].keypoints is not None:
            comp_keypoints.append([kp.cpu().numpy() for kp in comp_results[0].keypoints.data])
        else:
            comp_keypoints.append(None)
    
    print("\nPhase 2: Finding matching sequences...")
    # Find matching sequences
    matches = find_matching_sequences(ref_keypoints, 
                                   [kp[0] if kp else None for kp in comp_keypoints])
    
    if not matches:
        print("No matching sequences found!")
        return
    
    print(f"Found {len(matches)} matching sequences")
    
    # Reset video captures for visualization
    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    comp_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\nPhase 3: Generating comparison video...")
    frame_count = 0
    current_match = None
    
    while ref_cap.isOpened() and comp_cap.isOpened() and frame_count < max_frames:
        ref_ret, ref_frame = ref_cap.read()
        comp_ret, comp_frame = comp_cap.read()
        
        if not ref_ret or not comp_ret:
            break
            
        frame_count += 1
        print(f"\rProcessing frame {frame_count}", end="")
        
        # Find current matching sequence
        current_match = None
        for match in matches:
            ref_start, comp_start, length, similarity = match
            if ref_start <= frame_count < ref_start + length:
                current_match = match
                break
        
        # Create side-by-side view
        debug_frame = np.hstack((ref_frame, comp_frame))
        
        # Process current frame
        ref_results = model(ref_frame, verbose=False)
        comp_results = model(comp_frame, verbose=False)
        
        # Draw overlays and process comparison
        if len(ref_results) > 0 and len(comp_results) > 0:
            ref_result = ref_results[0]
            comp_result = comp_results[0]
            
            # Process reference pose
            if ref_result.keypoints is not None and len(ref_result.keypoints.data) > 0:
                ref_pose = ref_result.keypoints.data[0].cpu().numpy()
                
                # Draw reference skeleton
                draw_skeleton(debug_frame, ref_pose, color=COLORS['good'])
                
                # Draw reference bounding box
                if ref_result.boxes is not None and len(ref_result.boxes) > 0:
                    box = ref_result.boxes[0].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), COLORS['good'], 2)
                    cv2.putText(debug_frame, "Reference", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['good'], 2)
                
                # Process comparison poses
                if comp_result.keypoints is not None:
                    for dancer_idx, comp_pose in enumerate(comp_result.keypoints.data):
                        comp_pose = comp_pose.cpu().numpy()
                        
                        # Normalize poses for comparison
                        norm_ref, norm_comp, scaling_info = normalize_for_depth_comparison(
                            ref_pose, comp_pose
                        )
                        
                        if norm_ref is not None and norm_comp is not None:
                            # Get previous frames for velocity calculation
                            ref_prev = ref_keypoints[frame_count-1] if frame_count > 0 else None
                            comp_prev = (comp_keypoints[frame_count-1][dancer_idx] 
                                       if frame_count > 0 and comp_keypoints[frame_count-1] 
                                       else None)
                            
                            # Compare poses
                            difference, detailed_diffs = compare_poses(
                                norm_ref, norm_comp,
                                ref_prev, comp_prev
                            )
                            
                            # Choose color based on difference
                            if difference > 0.6:
                                color = COLORS['bad']
                            elif difference > 0.3:
                                color = COLORS['warning']
                            else:
                                color = COLORS['good']
                            
                            # Draw comparison skeleton
                            draw_skeleton(debug_frame, comp_pose, 
                                       offset_x=frame_width, color=color)
                            
                            # Draw comparison bounding box
                            if comp_result.boxes is not None and dancer_idx < len(comp_result.boxes):
                                box = comp_result.boxes[dancer_idx].xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, box)
                                x1 += frame_width
                                x2 += frame_width
                                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Add labels and scores
                                cv2.putText(debug_frame, f"Dancer {dancer_idx + 1}", 
                                          (x1, y1 - 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                cv2.putText(debug_frame, 
                                          f"Diff: {difference:.1%}", 
                                          (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # Show joint-specific differences
                                if difference > 0.3:  # Only show details for significant differences
                                    y_offset = y2 + 20
                                    for joint_type, diffs in detailed_diffs.items():
                                        for joint_name, diff_val in diffs.items():
                                            if diff_val > 0.3:
                                                text = f"{joint_type}: {joint_name}: {diff_val:.1%}"
                                                cv2.putText(debug_frame, text,
                                                          (x1, y_offset),
                                                          cv2.FONT_HERSHEY_SIMPLEX,
                                                          0.4, color, 1)
                                                y_offset += 15
        
        # Add frame information
        cv2.putText(debug_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text'], 2)
                   
        if current_match:
            ref_start, comp_start, length, similarity = current_match
            cv2.putText(debug_frame, 
                       f"Matching Sequence (Similarity: {similarity:.1%})", 
                       (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['good'], 2)
        
        # Show debug view
        cv2.imshow('Comparison View', debug_frame)
        out.write(debug_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    ref_cap.release()
    comp_cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save analysis data
    output_filename = f'dance_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    analysis_data = {
        'video_info': {
            'reference_video': reference_path,
            'comparison_video': comparison_path,
            'total_frames': frame_count,
            'fps': fps
        },
        'matching_sequences': [
            {
                'ref_start_frame': int(m[0]),
                'comp_start_frame': int(m[1]),
                'length': int(m[2]),
                'similarity': float(m[3])
            }
            for m in matches
        ]
    }
    
    with open(output_filename, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\n\nAnalysis complete!")
    print(f"Processed {frame_count} frames")
    print(f"Found {len(matches)} matching sequences")
    print(f"Analysis data saved to {output_filename}")

def draw_skeleton(frame, keypoints, offset_x=0, color=(0, 255, 0)):
    """
    Draw pose skeleton on frame
    
    Args:
        frame: OpenCV image to draw on
        keypoints: Numpy array of keypoints
        offset_x: Horizontal offset for drawing (for side-by-side view)
        color: BGR color tuple for skeleton
    """
    # Define skeleton connections
    skeleton = [
        (5, 7),   # Left shoulder to left elbow
        (7, 9),   # Left elbow to left wrist
        (6, 8),   # Right shoulder to right elbow
        (8, 10),  # Right elbow to right wrist
        (5, 6),   # Left shoulder to right shoulder
        (5, 11),  # Left shoulder to left hip
        (6, 12),  # Right shoulder to right hip
        (11, 12), # Left hip to right hip
        (11, 13), # Left hip to left knee
        (13, 15), # Left knee to left ankle
        (12, 14), # Right hip to right knee
        (14, 16)  # Right knee to right ankle
    ]
    
    # Draw skeleton lines
    for start_idx, end_idx in skeleton:
        if (start_idx < len(keypoints) and end_idx < len(keypoints)):
            start_pt = keypoints[start_idx]
            end_pt = keypoints[end_idx]
            
            if (len(start_pt) >= 2 and len(end_pt) >= 2 and
                all(x > 0 for x in start_pt[:2]) and 
                all(x > 0 for x in end_pt[:2])):
                
                start_pos = (int(start_pt[0]) + offset_x, int(start_pt[1]))
                end_pos = (int(end_pt[0]) + offset_x, int(end_pt[1]))
                cv2.line(frame, start_pos, end_pos, color, 2)
    
    # Draw keypoints
    for kp in keypoints:
        if len(kp) >= 2 and all(x > 0 for x in kp[:2]):
            x, y = int(kp[0]) + offset_x, int(kp[1])
            cv2.circle(frame, (x, y), 4, color, -1)
                                                        
def extract_audio_from_video(video_path):
    """
    Extract audio from video file and return the audio signal
    
    Args:
        video_path: Path to video file
    Returns:
        tuple of (audio_signal, sample_rate)
    """
    # Create temporary WAV file
    temp_dir = tempfile.gettempdir()
    temp_wav = os.path.join(temp_dir, 'temp_audio.wav')
    
    # Use ffmpeg to extract audio
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM format
        '-ar', '44100',  # Sample rate
        '-ac', '1',  # Mono
        '-y',  # Overwrite output file
        temp_wav
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        # Read the wav file
        sample_rate, audio_signal = wavfile.read(temp_wav)
        # Convert to float32 and normalize
        audio_signal = audio_signal.astype(np.float32) / np.iinfo(np.int16).max
        
        # Clean up temp file
        os.remove(temp_wav)
        
        return audio_signal, sample_rate
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        return None, None
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None

def compute_audio_features(audio_signal, sample_rate, frame_length=2048, hop_length=512):
    """
    Compute audio features for matching
    
    Args:
        audio_signal: Audio time series
        sample_rate: Sampling rate
        frame_length: Length of each frame in samples
        hop_length: Number of samples between frames
    Returns:
        Audio features (mel spectrogram)
    """
    try:
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_signal, 
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=128
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    except Exception as e:
        print(f"Error computing audio features: {e}")
        return None

def find_audio_offset(ref_audio, comp_audio, ref_sr, comp_sr, fps):
    """
    Find the offset between two audio signals
    
    Args:
        ref_audio: Reference audio signal
        comp_audio: Comparison audio signal
        ref_sr: Reference sample rate
        comp_sr: Comparison sample rate
        fps: Video frame rate
    Returns:
        Offset in frames
    """
    try:
        # Ensure same length
        min_len = min(len(ref_audio), len(comp_audio))
        ref_audio = ref_audio[:min_len]
        comp_audio = comp_audio[:min_len]
        
        # Compute cross-correlation
        correlation = signal.correlate(ref_audio, comp_audio, mode='full')
        
        # Find the peak correlation
        peak_idx = np.argmax(np.abs(correlation))
        
        # Calculate offset in samples
        offset_samples = peak_idx - (len(ref_audio) - 1)
        
        # Convert to frames
        offset_frames = int(offset_samples * fps / ref_sr)
        
        # Calculate correlation strength
        max_corr = np.max(np.abs(correlation))
        normalized_corr = max_corr / (np.sqrt(np.sum(ref_audio**2) * np.sum(comp_audio**2)))
        
        print(f"Audio sync confidence: {normalized_corr:.2%}")
        
        return offset_frames, normalized_corr
        
    except Exception as e:
        print(f"Error finding audio offset: {e}")
        return 0, 0

def synchronize_videos(reference_path, comparison_path):
    """
    Find synchronization offset between two videos using audio
    
    Args:
        reference_path: Path to reference video
        comparison_path: Path to comparison video
    Returns:
        Frame offset (how many frames to offset comparison video)
    """
    print("Extracting audio from videos...")
    
    # Extract audio from both videos
    ref_audio, ref_sr = extract_audio_from_video(reference_path)
    comp_audio, comp_sr = extract_audio_from_video(comparison_path)
    
    if ref_audio is None or comp_audio is None:
        print("Failed to extract audio from videos")
        return 0
    
    print("Computing audio features...")
    
    # Compute features
    ref_features = compute_audio_features(ref_audio, ref_sr)
    comp_features = compute_audio_features(comp_audio, comp_sr)
    
    if ref_features is None or comp_features is None:
        print("Failed to compute audio features")
        return 0
    
    # Get video frame rate
    cap = cv2.VideoCapture(reference_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    print("Finding synchronization offset...")
    
    # Find offset
    offset_frames, confidence = find_audio_offset(ref_audio, comp_audio, ref_sr, comp_sr, fps)
    
    if confidence < 0.3:  # Arbitrary threshold
        print("Warning: Low confidence in audio synchronization")
        return 0
    
    print(f"Found offset of {offset_frames} frames (confidence: {confidence:.2%})")
    return offset_frames

def create_synchronized_video(reference_path, comparison_path, offset):
    """
    Create synchronized videos while maintaining original audio
    
    Args:
        reference_path: Path to reference video
        comparison_path: Path to comparison video
        offset: Frame offset (positive means comparison starts later)
    
    Returns:
        Tuple of (new_reference_path, new_comparison_path)
    """
    # Get video properties
    ref_cap = cv2.VideoCapture(reference_path)
    comp_cap = cv2.VideoCapture(comparison_path)
    
    fps = int(ref_cap.get(cv2.CAP_PROP_FPS))
    ref_cap.release()
    comp_cap.release()
    
    # Calculate time offset in seconds
    time_offset = abs(offset) / fps
    
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()
    
    if offset > 0:
        # Comparison video starts later, trim reference video
        print("Trimming reference video to match comparison...")
        
        output_ref = str(Path(temp_dir) / 'synced_reference.mp4')
        new_comp_path = comparison_path  # Keep comparison as is
        
        # Trim reference video while maintaining original audio
        trim_command = [
            'ffmpeg',
            '-i', reference_path,
            '-ss', str(time_offset),  # Start time offset
            '-c:v', 'libx264',        # Video codec
            '-c:a', 'aac',            # Audio codec
            '-strict', 'experimental',
            '-y',                      # Overwrite output file
            output_ref
        ]
        
        try:
            subprocess.run(trim_command, check=True, capture_output=True)
            print("Reference video trimmed successfully")
            new_ref_path = output_ref
            
        except subprocess.CalledProcessError as e:
            print(f"Error trimming reference video: {e.stderr.decode()}")
            return reference_path, comparison_path
            
    else:
        # Reference video starts later, trim comparison video
        print("Trimming comparison video to match reference...")
        
        new_ref_path = reference_path  # Keep reference as is
        output_comp = str('synced_comparison.mp4')
        
        # Trim comparison video while maintaining original audio
        trim_command = [
            'ffmpeg',
            '-i', comparison_path,
            '-ss', str(time_offset),  # Start time offset
            '-c:v', 'libx264',        # Video codec
            '-c:a', 'aac',            # Audio codec
            '-strict', 'experimental',
            '-y',                      # Overwrite output file
            output_comp
        ]
        
        try:
            subprocess.run(trim_command, check=True, capture_output=True)
            print("Comparison video trimmed successfully")
            new_comp_path = output_comp
            
        except subprocess.CalledProcessError as e:
            print(f"Error trimming comparison video: {e.stderr.decode()}")
            return reference_path, comparison_path
    
    return new_ref_path, new_comp_path

def verify_sync(reference_path, comparison_path):
    """
    Verify that videos are properly synchronized
    
    Args:
        reference_path: Path to reference video
        comparison_path: Path to comparison video
    Returns:
        bool: True if videos appear synchronized
    """
    ref_cap = cv2.VideoCapture(reference_path)
    comp_cap = cv2.VideoCapture(comparison_path)
    
    # Get duration of both videos
    ref_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    comp_frames = int(comp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_fps = int(ref_cap.get(cv2.CAP_PROP_FPS))
    comp_fps = int(comp_cap.get(cv2.CAP_PROP_FPS))
    
    ref_duration = ref_frames / ref_fps
    comp_duration = comp_frames / comp_fps
    
    # Check if durations are similar (within 1 second)
    duration_diff = abs(ref_duration - comp_duration)
    
    print(f"\nSync Verification:")
    print(f"Reference duration: {ref_duration:.2f}s")
    print(f"Comparison duration: {comp_duration:.2f}s")
    print(f"Duration difference: {duration_diff:.2f}s")
    
    ref_cap.release()
    comp_cap.release()
    
    return duration_diff < 1.0

def main_sync_process(reference_path, comparison_path):
    """
    Main function to handle video synchronization process
    
    Args:
        reference_path: Path to reference video
        comparison_path: Path to comparison video
    Returns:
        Tuple of synchronized video paths
    """
    print("Step 1: Detecting audio synchronization...")
    offset = synchronize_videos(reference_path, comparison_path)
    
    if offset == 0:
        print("No synchronization needed")
        return reference_path, comparison_path
    
    print(f"\nStep 2: Creating synchronized videos...")
    new_ref_path, new_comp_path = create_synchronized_video(
        reference_path, comparison_path, offset
    )
    
    print("\nStep 3: Verifying synchronization...")
    if verify_sync(new_ref_path, new_comp_path):
        print("Videos successfully synchronized!")
    else:
        print("Warning: Videos may not be perfectly synchronized")
    
    return new_ref_path, new_comp_path

def test_sync():
    """
    Test the synchronization process
    """
    reference_path = "reference.mp4"
    comparison_path = "compare.mp4"
    
    synced_ref, synced_comp = main_sync_process(reference_path, comparison_path)
    print(f"\nSynchronized videos created:")
    print(f"Reference: {synced_ref}")
    print(f"Comparison: {synced_comp}")
    
test_sync()                                                                                                           
#compare_dance_videos('./reference.mp4','./compare.mp4')