import asyncio

import pytest
from fastcs.attributes import Attribute
from pytest_mock import MockerFixture

from fastcs_eiger.eiger_controller import (
    IGNORED_KEYS,
    MISSING_KEYS,
    EigerController,
    EigerDetectorController,
    EigerSubsystemController,
    Subsystem,
)

_lock = asyncio.Lock()


@pytest.mark.asyncio
async def test_subsystem(mock_connection, detector_config_keys, detector_status_keys):
    subsystem = Subsystem("detector", _lock)
    parameters = await subsystem.introspect(mock_connection)
    assert all(parameter.key not in IGNORED_KEYS for parameter in parameters)
    for parameter in parameters:
        assert parameter.key not in IGNORED_KEYS
        if parameter.mode == "config":
            assert (
                parameter.key in detector_config_keys
                or parameter.key in MISSING_KEYS["detector"]["config"]
            )
        elif parameter.mode == "status":
            assert (
                parameter.key in detector_status_keys
                or parameter.key in MISSING_KEYS["detector"]["status"]
            )

    # test queue_update side effect
    assert not subsystem.stale
    await subsystem.queue_update(["chi_start"])
    assert subsystem.parameter_updates == {"chi_start"}
    assert subsystem.stale


@pytest.mark.asyncio
async def test_subsystem_controller_initialises(mock_connection):
    subsystem = Subsystem("stream", _lock)
    subsystem_controller = EigerSubsystemController(subsystem, mock_connection)
    await subsystem_controller.initialise()


@pytest.mark.asyncio
async def test_detector_subsystem_controller(mock_connection):
    subsystem = Subsystem("detector", _lock)
    subsystem_controller = EigerDetectorController(subsystem, mock_connection)
    await subsystem_controller.initialise()

    for attr_name in dir(subsystem_controller):
        attr = getattr(subsystem_controller, attr_name)
        if isinstance(attr, Attribute):
            assert "threshold" not in attr_name
        # then test it is in threshold sub controller
    threshold_controller = subsystem_controller.get_sub_controllers().get(
        "THRESHOLD", None
    )
    assert threshold_controller
    for attr_name in dir(threshold_controller):
        attr = getattr(threshold_controller, attr_name)
        if isinstance(attr, Attribute):
            assert "threshold" in attr._updater.uri


@pytest.mark.asyncio
async def test_eiger_controller_initialises(mocker: MockerFixture, mock_connection):
    eiger_controller = EigerController("127.0.0.1", 80)
    connection = mocker.patch.object(eiger_controller, "connection")
    connection.get = mock_connection.get
    await eiger_controller.initialise()
    assert list(eiger_controller.get_sub_controllers().keys()) == [
        "DETECTOR",
        "STREAM",
        "MONITOR",
    ]
    connection.get.assert_any_call("detector/api/1.8.0/status/state")
    connection.get.assert_any_call("stream/api/1.8.0/status/state")
    connection.get.assert_any_call("monitor/api/1.8.0/status/state")
