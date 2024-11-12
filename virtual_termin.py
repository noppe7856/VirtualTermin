import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import sys
import simpleaudio as sa
import threading
from scipy import signal

# ログの設定
# ロギングの設定を行う関数（ログレベル、フォーマット、出力先の設定）
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# 音再生のロックを設定（同時に複数の音が鳴らないように）
sound_lock = threading.Lock()

# 基本周波数の設定（A4 = 440Hz）
base_frequency = 440  # A4の音の周波数（ヘルツ）

# 音を再生するための最小半径の設定
min_radius_to_play_sound = 20  # 音を鳴らすための最小の球の半径

# 音再生スレッドを継続的に実行するためのフラグ
global sound_playing
sound_playing = False

# 音を再生する関数
# 指定された周波数の音を再生し続ける

def continuous_play_sound():
    global sound_playing
    play_obj = None
    sample_rate = 44100  # サンプルレート（Hz）
    while True:
        if sound_playing:
            frequency = base_frequency
            t = np.linspace(0, 0.5, int(sample_rate * 0.5), False)  # 0.5秒間の音生成
            wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # 正弦波を生成
            audio = (wave * 32767).astype(np.int16)  # 16ビットPCMに変換
            try:
                with sound_lock:
                    if play_obj is None or not play_obj.is_playing():
                        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
            except Exception as e:
                logging.error(f"音の再生中にエラーが発生しました: {e}")
                sys.stdout.flush()
        else:
            if play_obj is not None and play_obj.is_playing():
                play_obj.stop()
                play_obj = None
        time.sleep(0.1)  # 音のチェック間隔

# 音再生スレッドを開始
threading.Thread(target=continuous_play_sound, daemon=True).start()

# ビデオキャプチャを行う関数
# Webカメラからの映像をキャプチャし、手の検出と描画を行う
def capture_video():
    cap = cv2.VideoCapture(0)  # カメラのキャプチャを開始
    retry_attempts = 3  # カメラの再接続試行回数
    retry_count = 0
    last_frame_time = time.time()
    frame_timeout = 5  # 秒数を指定、5秒間フレームが取得できなければ再試行

    # MediaPipe Handsの初期化
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            current_time = time.time()
            if not success:
                handle_frame_failure(cap, current_time, last_frame_time, retry_count, retry_attempts, frame_timeout)
                continue

            # フレームの取得に成功した場合の処理
            last_frame_time, retry_count = handle_successful_frame(current_time, last_frame_time, retry_count)

            # 画像を反転（鏡のように表示するため）
            image = cv2.flip(image, 1)
            # フレームの処理を行う（手のランドマーク検出など）
            process_frame(image, hands)

            # ウィンドウに画像を表示
            cv2.imshow('Hand Tracking', image)
            time.sleep(0.02)  # フレームレート調整のためにスリープを追加し、CPU使用率を抑える

            # 'q'キーが押されたらループを終了
            if cv2.waitKey(5) & 0xFF == ord('q'):
                logging.info("プログラムを終了します")
                sys.stdout.flush()
                break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()
    logging.info("リソースを解放しました")
    sys.stdout.flush()

# フレーム取得失敗時の処理を行う関数
# カメラの再接続や終了を制御
def handle_frame_failure(cap, current_time, last_frame_time, retry_count, retry_attempts, frame_timeout):
    logging.warning("カメラからのフレーム取得に失敗しました")
    sys.stdout.flush()
    # フレーム取得がタイムアウトした場合の処理
    if current_time - last_frame_time > frame_timeout:
        retry_count += 1
        logging.error(f"フレームの取得がタイムアウトしました。再試行します（{retry_count}/{retry_attempts}回目）")
        sys.stdout.flush()
        cap.release()
        if retry_count <= retry_attempts:
            cap = cv2.VideoCapture(0)  # カメラの再接続を試行
            last_frame_time = current_time
            time.sleep(1)  # 少し待ってから再接続
        else:
            # 最大試行回数に達した場合、プログラムを終了
            logging.critical("カメラの再接続が最大試行回数に達しました。プログラムを終了します。")
            sys.stdout.flush()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

# フレームの取得に成功した場合の処理
# タイムスタンプを更新し、リトライカウントをリセット
def handle_successful_frame(current_time, last_frame_time, retry_count):
    last_frame_time = current_time
    retry_count = 0
    logging.debug("フレームを正常に取得しました")  # ログレベルをDEBUGに変更して頻度を下げる
    sys.stdout.flush()
    return last_frame_time, retry_count

# フレームの処理を行う関数
# フレームをRGBに変換し、手のランドマークを検出して描画
def process_frame(image, hands):
    # BGR画像をRGBに変換（MediaPipeはRGBを使用するため）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 手の検出を行う
    results = hands.process(image_rgb)
    # 検出された手のランドマークを描画
    if results.multi_hand_landmarks:
        logging.info("手のランドマークを検出しました")
        sys.stdout.flush()
        for hand_landmarks in results.multi_hand_landmarks:
            draw_hand_landmarks(image, hand_landmarks)
            handle_hand_landmarks(image, hand_landmarks)

# 手のランドマークを描画する関数
# MediaPipeの描画ユーティリティを使用して手のランドマークを描画
def draw_hand_landmarks(image, hand_landmarks):
    mp_drawing.draw_landmarks(
        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # ランドマークを描画する設定
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # 接続線を描画する設定
    )

# 手のランドマークを処理する関数
# 親指、人差し指、中指の先端のランドマークを使って重心と最大距離を計算し、必要に応じて音を再生または停止

def handle_hand_landmarks(image, hand_landmarks):
    global sound_playing

    # 親指、人差し指、中指の先のランドマークを取得
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # 画像上の座標に変換（ピクセル単位）
    h, w, _ = image.shape
    thumb_tip_coords = np.array([int(thumb_tip.x * w), int(thumb_tip.y * h), 0])
    index_tip_coords = np.array([int(index_tip.x * w), int(index_tip.y * h), 0])
    middle_tip_coords = np.array([int(middle_tip.x * w), int(middle_tip.y * h), 0])

    # 親指、人差し指、中指の先の重心を計算
    center_coords = (thumb_tip_coords + index_tip_coords + middle_tip_coords) // 3
    center_coords = tuple(center_coords[:2])

    # 親指、人差し指、中指の先の最大距離を計算
    distances = [
        np.linalg.norm(thumb_tip_coords - index_tip_coords),
        np.linalg.norm(thumb_tip_coords - middle_tip_coords),
        np.linalg.norm(index_tip_coords - middle_tip_coords)
    ]
    max_distance = int(max(distances))

    logging.info(f"最大距離: {max_distance}, 重心座標: {center_coords}")
    sys.stdout.flush()

    # 親指、人差し指、中指の先に接する球を描画（描画を簡素化）
    radius = max_distance // 2
    cv2.circle(image, center_coords, radius, (0, 255, 0), 2)

    # 一定のサイズ以上で音を鳴らす（音の再生を別スレッドで実行）
    if radius >= min_radius_to_play_sound:
        sound_playing = True
    else:
        sound_playing = False

# MediaPipe Handsの初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# メインプログラムの実行
def main():
    setup_logging()
    capture_video()

if __name__ == "__main__":
    main()
