#!/usr/bin/env python

from typing import Optional, List, Dict, Any
import datetime
import logging
import json
import os
import os.path
import pathlib
import pprint
import shutil
import sys
import time

import ring_doorbell
from oauthlib.oauth2 import MissingTokenError
import requests
import retry
import twilio.rest

logger = logging.getLogger("ring_get_images")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class CouldNotGetRingOtpCodeException(Exception):
    pass


class TwilioWrapper:
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str

    def __init__(
        self, twilio_account_sid: str, twilio_auth_token: str, twilio_phone_number: str
    ) -> None:
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_phone_number = twilio_phone_number

    def delete_old_messages(self) -> None:
        client = twilio.rest.Client(self.twilio_account_sid, self.twilio_auth_token)
        date_threshold: datetime.date = datetime.date.today() - datetime.timedelta(
            days=1
        )
        for sms in client.messages.stream(
            date_sent_before=date_threshold, to=self.twilio_phone_number
        ):
            logger.info("delete_old_messages - deleting SMS with ID %s", sms.sid)
            client.messages.delete(sms.sid)

    def wait_for_ring_otp_code(
        self, timeout_seconds: int = 60, interval_seconds: int = 5
    ) -> Optional[str]:
        logger.info("wait_for_ring_otp_code entry")
        start: datetime.datetime = datetime.datetime.now()
        while True:
            if datetime.datetime.now() - start > datetime.timedelta(
                seconds=timeout_seconds
            ):
                logger.info(
                    "wait_for_ring_otp_code - timed out waiting for ring OTP code"
                )
                break
            maybe_otp_code: Optional[str] = self.get_last_ring_otp_code()
            if maybe_otp_code is not None:
                logger.info("wait_for_ring_otp_code - found Ring OTP code")
                return maybe_otp_code
            logger.info("wait_for_ring_otp_code - sleeping and waiting...")
            time.sleep(interval_seconds)

        return None

    def get_last_ring_otp_code(self, max_sms_age_seconds: int = 300) -> Optional[str]:
        logger.info("get_last_ring_otp_code entry")
        client = twilio.rest.Client(self.twilio_account_sid, self.twilio_auth_token)
        for sms in client.messages.stream(to=self.twilio_phone_number):
            logger.info("Twilio SMS SID: %s", sms.sid)
            age: datetime.timedelta = (
                datetime.datetime.now(datetime.timezone.utc) - sms.date_created
            )
            if age > datetime.timedelta(seconds=max_sms_age_seconds):
                logger.info("message too old, could not find Ring OTP code")
                break
            if not sms.body.startswith("Your Ring verification code is:"):
                logger.info("skipping SMS")
                continue
            code: str = sms.body.partition("Your Ring verification code is:")[
                -1
            ].strip()
            return code

        return None


class RingWrapper:
    ring_username: str
    ring_password: str
    twilio_wrapper: TwilioWrapper

    def __init__(
        self,
        ring_username: str,
        ring_password: str,
        twilio_wrapper: TwilioWrapper,
        sleep_interval_seconds: int = 5,
    ):
        self.ring_username = ring_username
        self.ring_password = ring_password
        self.twilio_wrapper = twilio_wrapper
        self.sleep_interval_seconds = sleep_interval_seconds

    def login(self) -> None:
        auth = ring_doorbell.Auth("CameraPackageNotifier/1.0", None, None)
        try:
            logger.info("main - logging in without OTP code...")
            auth.fetch_token(self.ring_username, self.ring_password)
        except MissingTokenError:
            logger.info("main - missing token error, logging in with OTP code...")

            # TODO this doesn't work!
            # self.twilio_wrapper.delete_old_messages()

            otp_code: Optional[str] = self.twilio_wrapper.wait_for_ring_otp_code()
            if otp_code is None:
                logger.error("main - could not get Ring OTP code")
                raise CouldNotGetRingOtpCodeException()
            time.sleep(self.sleep_interval_seconds)
            auth.fetch_token(self.ring_username, self.ring_password, otp_code)

        self.ring = ring_doorbell.Ring(auth)
        time.sleep(self.sleep_interval_seconds)
        self.ring.update_devices()
        time.sleep(self.sleep_interval_seconds)
        logger.info("RingWrapper - devices: %s", self.ring.devices())

    def get_device(self, device_name: str) -> ring_doorbell.RingDoorBell:
        return [
            device
            for device in self.ring.devices()["stickup_cams"]
            + self.ring.devices()["doorbots"]
            if device.name == device_name
        ][0]

    def get_device_history(
        self,
        device_name: str,
        history_limit: int = 30,
        older_than_event_id: Optional[str] = None,
        ring_event_page_limit: int = 30,
    ) -> List[Dict[str, Any]]:
        device: ring_doorbell.RingDoorBell = self.get_device(device_name)

        logger.info("Getting device history...")
        events: List[Dict[str, Any]] = []
        while True:
            if len(events) == 0:
                logger.info("main - getting events for first time...")
                if older_than_event_id is None:
                    current_events = device.history(
                        limit=ring_event_page_limit, enforce_limit=True
                    )
                else:
                    current_events = device.history(
                        limit=ring_event_page_limit,
                        enforce_limit=True,
                        older_than=older_than_event_id,
                    )
            else:
                logger.info("main - getting events older than %s", events[-1]["id"])
                current_events = device.history(
                    limit=ring_event_page_limit,
                    enforce_limit=True,
                    older_than=events[-1]["id"],
                )
            time.sleep(self.sleep_interval_seconds)
            if len(current_events) == 0:
                logger.info("main - no more events")
                break
            events.extend(
                event for event in current_events if event["kind"] != "on_demand_link"
            )
            if len(events) >= history_limit:
                logger.info("main - got up to event limit")
                break

        return events

    @retry.retry(tries=5, delay=10)
    def get_event_recording_url(
        self, device: ring_doorbell.RingDoorBell, event_id: str
    ) -> Any:
        logger.info("get_event_recording_url for event_id %s" % (event_id,))
        return device.recording_url(event_id)

    def download_event(
        self, device_name: str, event_id: str, destination_path: pathlib.Path
    ) -> pathlib.Path:
        logger.info("Getting event %s URL..." % (event_id,))

        video_path: pathlib.Path = destination_path / "video.mp4"
        if video_path.exists():
            logger.info("video path %s already exists, skipping" % (video_path,))
            return video_path

        device: ring_doorbell.RingDoorBell = self.get_device(device_name)
        time.sleep(self.sleep_interval_seconds)
        url: str = self.get_event_recording_url(device, event_id)

        logger.info("Downloading event...")
        return self.download_file(url, video_path)

    def download_file(self, url: str, local_path: pathlib.Path) -> pathlib.Path:
        with requests.get(url, stream=True) as r:
            with local_path.open("wb") as f:
                shutil.copyfileobj(r.raw, f)
        return local_path


def main(
    destination_path: pathlib.Path,
    sleep_interval_seconds: int = 10,
    event_limit: int = 400,
    ring_event_page_limit: int = 30,
) -> None:
    ring_username: str = os.environ["RING_USERNAME"]
    ring_password: str = os.environ["RING_PASSWORD"]
    twilio_account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    twilio_auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    twilio_phone_number = os.environ["TWILIO_PHONE_NUMBER"]

    twilio_wrapper: TwilioWrapper = TwilioWrapper(
        twilio_account_sid, twilio_auth_token, twilio_phone_number
    )
    ring_wrapper: RingWrapper = RingWrapper(
        ring_username, ring_password, twilio_wrapper, sleep_interval_seconds,
    )
    ring_wrapper.login()
    device_name: str = "Front Door Cam"
    events: List[Dict[str, Any]] = ring_wrapper.get_device_history(
        device_name,
        history_limit=event_limit,
        ring_event_page_limit=ring_event_page_limit,
    )
    for event in events:
        specific_destination_path: pathlib.Path = destination_path / str(event["id"])
        if not specific_destination_path.exists():
            os.mkdir(specific_destination_path)

        event_info_file: pathlib.Path = specific_destination_path / "event_info"
        flag_file: pathlib.Path = specific_destination_path / "successful"

        event_info_file.write_text(
            json.dumps(event, indent=4, sort_keys=True, default=str)
        )
        if flag_file.exists():
            logger.info("event id %s already exists" % (event["id"],))
            continue

        shutil.rmtree(specific_destination_path.absolute())
        os.mkdir(specific_destination_path.absolute())
        event_info_file.write_text(
            json.dumps(event, indent=4, sort_keys=True, default=str)
        )
        video_path: pathlib.Path = ring_wrapper.download_event(
            device_name,
            event_id=event["id"],
            destination_path=specific_destination_path,
        )

        flag_file.touch()


if __name__ == "__main__":
    destination_path: pathlib.Path = pathlib.Path(sys.argv[1])
    main(destination_path)
