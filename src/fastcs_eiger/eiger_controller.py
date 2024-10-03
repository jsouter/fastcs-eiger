import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Literal

import numpy as np
from fastcs.attributes import Attribute, AttrR, AttrRW, AttrW
from fastcs.controller import Controller, SubController
from fastcs.datatypes import Bool, Float, Int, String
from fastcs.wrappers import command, scan
from PIL import Image

from fastcs_eiger.http_connection import HTTPConnection, HTTPRequestError
from fastcs_eiger.util import partition

# Keys to be ignored when introspecting the detector to create parameters
IGNORED_KEYS = [
    # Big arrays
    "countrate_correction_table",
    "pixel_mask",
    "threshold/1/pixel_mask",
    "threshold/2/pixel_mask",
    "flatfield",
    "threshold/1/flatfield",
    "threshold/2/flatfield",
    # Deprecated
    "board_000/th0_humidity",
    "board_000/th0_temp",
    # TODO: Value is [value, max], rather than using max metadata
    "buffer_fill_level",
    # TODO: Handle array values
    "detector_orientation",
    "detector_translation",
    # TODO: Is it a bad idea to include these?
    "test_image_mode",
    "test_image_value",
]

# Parameters that are in the API but missing from keys
MISSING_KEYS: dict[str, dict[str, list[str]]] = {
    "detector": {"status": ["error"], "config": ["wavelength"]},
    "monitor": {"status": [], "config": []},
    "stream": {"status": ["error"], "config": []},
}


def command_uri(key: str) -> str:
    return f"detector/api/1.8.0/command/{key}"


def detector_command(fn) -> Any:
    return command(group="DetectorCommand")(fn)


@dataclass
class EigerHandler:
    """
    Handler for FastCS Attribute Creation

    Dataclass that is called using the AttrR, AttrRW function.
    Handler uses uri of detector to collect data for PVs
    """

    uri: str
    update_period: float = 0.2

    async def put(
        self,
        controller: "EigerSubsystemController | EigerSubController",
        _: AttrW,
        value: Any,
    ) -> None:
        parameters_to_update = await controller.connection.put(self.uri, value)
        if not parameters_to_update:
            parameters_to_update = [self.uri.split("/", 4)[-1]]
            print(f"Manually fetching parameter {parameters_to_update}")
        elif "difference_mode" in parameters_to_update:
            parameters_to_update[parameters_to_update.index("difference_mode")] = (
                "threshold/difference/mode"
            )
            print(
                f"Fetching parameters after setting {self.uri}: {parameters_to_update},"
                " replacing incorrect key 'difference_mode'"
            )
        else:
            print(
                f"Fetching parameters after setting {self.uri}: {parameters_to_update}"
            )

        await controller.subsystem.queue_update(parameters_to_update)

    async def update(self, controller: "EigerController", attr: AttrR) -> None:
        try:
            response = await controller.connection.get(self.uri)
            await attr.set(response["value"])
        except Exception as e:
            print(f"Failed to get {self.uri}:\n{e.__class__.__name__} {e}")


class EigerConfigHandler(EigerHandler):
    """Handler for config parameters that are polled once on startup."""

    first_poll_complete: bool = False

    async def update(self, controller: "EigerController", attr: AttrR) -> None:
        # Only poll once on startup
        if not self.first_poll_complete:
            await super().update(controller, attr)
            if isinstance(attr, AttrRW):
                # Sync readback value to demand
                await attr.update_display_without_process(attr.get())

            self.first_poll_complete = True

    async def config_update(self, controller: "EigerController", attr: AttrR) -> None:
        await super().update(controller, attr)


@dataclass
class LogicHandler:
    """
    Handler for FastCS Attribute Creation

    Dataclass that is called using the AttrR, AttrRW function.
    Used for dynamically created attributes that are added for additional logic
    """

    async def put(self, _: "EigerController", attr: AttrW, value: Any) -> None:
        await attr.set(value)


EIGER_HANDLERS: dict[str, type[EigerHandler]] = {
    "status": EigerHandler,
    "config": EigerConfigHandler,
}


@dataclass
class EigerParameter:
    key: str
    """Last section of URI within a subsystem/mode."""
    subsystem: Literal["detector", "stream", "monitor"]
    """Subsystem within detector API."""
    mode: Literal["status", "config"]
    """Mode of parameter within subsystem."""
    response: dict[str, Any]
    """JSON response from GET of parameter."""

    @property
    def uri(self) -> str:
        """Full URI for HTTP requests."""
        return f"{self.subsystem}/api/1.8.0/{self.mode}/{self.key}"


EIGER_PARAMETER_SUBSYSTEMS = EigerParameter.__annotations__["subsystem"].__args__
EIGER_PARAMETER_MODES = EigerParameter.__annotations__["mode"].__args__


# Flatten nested uri keys - e.g. threshold/1/mode -> threshold_1_mode
def _key_to_attribute_name(key: str):
    return key.replace("/", "_")


class EigerController(Controller):
    """
    Controller Class for Eiger Detector

    Used for dynamic creation of variables useed in logic of the EigerFastCS backend.
    Sets up all connections with the Simplon API to send and receive information
    """

    # Detector parameters to use in internal logic
    trigger_mode = AttrRW(String())  # TODO: Include URI and validate type from API

    # Internal Attributes
    stale_parameters = AttrR(Bool())
    trigger_exposure = AttrRW(Float(), handler=LogicHandler())

    def __init__(self, ip: str, port: int) -> None:
        super().__init__()
        self._ip = ip
        self._port = port
        self.connection = HTTPConnection(self._ip, self._port)
        self.subsystems: dict[str, Subsystem] = {}
        # Parameter update logic
        self.parameter_update_lock = asyncio.Lock()

    async def initialise(self) -> None:
        """Create attributes by introspecting detector.

        The detector will be initialized if it is not already.

        """
        self.connection.open()

        # Check current state of detector_state to see if initializing is required.
        state_val = await self.connection.get("detector/api/1.8.0/status/state")
        if state_val["value"] == "na":
            print("Initializing Detector")
            await self.initialize()

        try:
            for subsystem_name in EIGER_PARAMETER_SUBSYSTEMS:
                subsystem = Subsystem(subsystem_name, self.parameter_update_lock)
                self.subsystems[subsystem_name] = subsystem
                subsystem_controller_cls = (
                    EigerDetectorController
                    if subsystem_name == "detector"
                    else EigerSubsystemController
                )
                controller = subsystem_controller_cls(subsystem, self.connection)
                self.register_sub_controller(subsystem_name.upper(), controller)
                await controller.initialise()
        except HTTPRequestError:
            print("\nAn HTTP request failed while introspecting detector:\n")
            raise

    @detector_command
    async def initialize(self):
        await self.connection.put(command_uri("initialize"))

    @detector_command
    async def arm(self):
        await self.connection.put(command_uri("arm"))

    @detector_command
    async def trigger(self):
        match self.trigger_mode.get(), self.trigger_exposure.get():
            case ("inte", exposure) if exposure > 0.0:
                await self.connection.put(command_uri("trigger"), exposure)
            case ("ints" | "inte", _):
                await self.connection.put(command_uri("trigger"))
            case _:
                raise RuntimeError("Can only do soft trigger in 'ints' or 'inte' mode")

    @detector_command
    async def disarm(self):
        await self.connection.put(command_uri("disarm"))

    @detector_command
    async def abort(self):
        await self.connection.put(command_uri("abort"))

    @detector_command
    async def cancel(self):
        await self.connection.put(command_uri("cancel"))

    @scan(0.1)
    async def update(self):
        """Periodically check for parameters that need updating from the detector."""
        await self.stale_parameters.set(
            any(c.stale_parameters.get() for c in self.get_sub_controllers().values())
        )
        controller_updates = [c.update() for c in self.get_sub_controllers().values()]
        await asyncio.gather(*controller_updates)

    @scan(1)
    async def handle_monitor(self):
        """Poll monitor images to display."""
        response, image_bytes = await self.connection.get_bytes(
            "monitor/api/1.8.0/images/next"
        )
        if response.status != 200:
            return
        else:
            image = Image.open(BytesIO(image_bytes))

            # TODO: Populate waveform PV to display as image, once supported in PVI
            print(np.array(image))


def _create_attributes(
    parameters: list[EigerParameter],
    attr_namer: Callable[[str], str] | None = None,
    group_namer: Callable[[EigerParameter], str] | None = None,
):
    """Create ``Attribute``s from ``EigerParameter``s.

    Args:
        parameters: ``EigerParameter``s to create ``Attributes`` from

    """
    attributes: dict[str, Attribute] = {}
    for parameter in parameters:
        group = group_namer(parameter) if group_namer else parameter.mode.capitalize()
        match parameter.response["value_type"]:
            case "float":
                datatype = Float()
            case "int" | "uint":
                datatype = Int()
            case "bool":
                datatype = Bool()
            case "string" | "datetime" | "State" | "string[]":
                datatype = String()
            case _:
                print(f"Failed to handle {parameter}")

        attribute_name = attr_namer(parameter.key) if attr_namer else parameter.key

        match parameter.response["access_mode"]:
            case "r":
                attributes[attribute_name] = AttrR(
                    datatype,
                    handler=EIGER_HANDLERS[parameter.mode](parameter.uri),
                    group=group,
                )
            case "rw":
                attributes[attribute_name] = AttrRW(
                    datatype,
                    handler=EIGER_HANDLERS[parameter.mode](parameter.uri),
                    group=group,
                    allowed_values=parameter.response.get("allowed_values", None),
                )

    return attributes


class Subsystem:
    def __init__(
        self, name: Literal["detector", "stream", "monitor"], lock: asyncio.Lock
    ):
        self.name = name
        self.parameter_update_lock = lock
        self.parameter_updates: set[str] = set()
        self.stale = False
        self.attribute_mapping: dict[str, AttrR] = {}

    async def introspect_detector_subsystem(
        self, connection: HTTPConnection
    ) -> list[EigerParameter]:
        parameters = []
        for mode in EIGER_PARAMETER_MODES:
            subsystem_keys = [
                parameter
                for parameter in await connection.get(
                    f"{self.name}/api/1.8.0/{mode}/keys"
                )
                if parameter not in IGNORED_KEYS
            ] + MISSING_KEYS[self.name][mode]
            requests = [
                connection.get(f"{self.name}/api/1.8.0/{mode}/{key}")
                for key in subsystem_keys
            ]
            responses = await asyncio.gather(*requests)

            parameters.extend(
                [
                    EigerParameter(
                        key=key, subsystem=self.name, mode=mode, response=response
                    )
                    for key, response in zip(subsystem_keys, responses, strict=False)
                ]
            )

        return parameters

    async def queue_update(self, parameters: list[str]):
        """Add the given parameters to the list of parameters to update.

        Args:
            parameters: Parameters to be updated

        """
        async with self.parameter_update_lock:
            for parameter in parameters:
                self.parameter_updates.add(parameter)

            self.stale = True


class EigerSubsystemController(SubController):
    stale_parameters = AttrR(Bool())
    _subcontroller_mapping: dict[str, "EigerSubController"]

    def __init__(self, subsystem: Subsystem, connection: HTTPConnection):
        self.subsystem = subsystem
        self.connection = connection
        self._subcontroller_mapping = {}
        super().__init__()

    async def initialise(self) -> None:
        parameters = await self.subsystem.introspect_detector_subsystem(self.connection)
        await self._create_subcontrollers(parameters)
        attributes = _create_attributes(parameters, _key_to_attribute_name)

        for name, attribute in attributes.items():
            setattr(self, name, attribute)

    async def update(self):
        if not self.subsystem.parameter_updates:
            if self.subsystem.stale:
                self.subsystem.stale = False

        await self.stale_parameters.set(self.subsystem.stale)

        async with self.subsystem.parameter_update_lock:
            parameters = self.subsystem.parameter_updates.copy()
            self.subsystem.parameter_updates.clear()

        # Release lock while fetching parameters - this may be slow
        parameter_updates: list[Coroutine] = []
        for parameter in parameters:
            if parameter in IGNORED_KEYS:
                continue
            match self._get_attribute(parameter):
                # TODO: mypy doesn't understand AttrR as a type for some reason:
                # `error: Expected type in class pattern; found "Any"  [misc]`
                case AttrR(updater=EigerConfigHandler() as updater) as attr:  # type: ignore [misc]
                    parameter_updates.append(updater.config_update(self, attr))
                case _ as attr:
                    print(f"Failed to handle update for {parameter}: {attr}")
        await asyncio.gather(*parameter_updates)

    async def _create_subcontrollers(self, parameters: list[EigerParameter]):
        """Create and register subcontrollers to logically group attributes within
        the subsystem.

        Args:
            parameters: list of ``EigerParameter``s to be filtered and passed into
            ``EigerSubController``s. ``EigerParameter``s wich are filtered should be
            removed from the list.

        """

    def _get_attribute(self, key: str) -> AttrR | None:
        # attributes with non-standard names or belong to sub controllers get added to
        # the attribute map for the subsystem
        if key in self.subsystem.attribute_mapping:
            return self.subsystem.attribute_mapping[key]

        # if not in mapping, get from this controller
        attr_name = _key_to_attribute_name(key)
        return getattr(self, attr_name, None)


class EigerDetectorController(EigerSubsystemController):
    async def _create_subcontrollers(self, parameters: list[EigerParameter]):
        def __threshold_parameter(parameter: EigerParameter):
            return "threshold" in parameter.key

        threshold_parameters, parameters[:] = partition(
            parameters, __threshold_parameter
        )

        threshold_controller = EigerThresholdController(
            threshold_parameters, self.connection, self.subsystem
        )
        self.register_sub_controller("THRESHOLD", threshold_controller)
        await threshold_controller.initialise()

        for parameter in threshold_parameters:
            self._subcontroller_mapping[parameter.key] = threshold_controller


class EigerSubController(SubController):  # for smaller parts of subsystems
    def __init__(
        self,
        parameters: list[EigerParameter],
        connection: HTTPConnection,
        subsystem: Subsystem,
    ):
        self._parameters = parameters
        self.subsystem = subsystem
        self.connection = connection
        super().__init__()

    async def initialise(self):
        attributes = _create_attributes(self._parameters, _key_to_attribute_name)
        for name, attribute in attributes.items():
            setattr(self, name, attribute)


class EigerThresholdController(EigerSubController):
    async def initialise(self):
        def __is_index(parameter: EigerParameter):
            parts = parameter.key.split("/")
            return len(parts) == 3 and parts[1].isnumeric()

        index_parameters, other_parameters = partition(self._parameters, __is_index)

        def __idx_group_name(parameter: EigerParameter) -> str:
            return "Threshold" + parameter.key.split("/")[1]

        index_attributes = _create_attributes(
            index_parameters, group_namer=__idx_group_name
        )
        for key, attribute in index_attributes.items():
            _, index, field = key.split("/")
            attr_name = f"{field}_{index}"
            setattr(self, attr_name, attribute)
            self.subsystem.attribute_mapping[key] = attribute

        other_attributes = _create_attributes(other_parameters)
        for key, attribute in other_attributes.items():
            attr_name = _key_to_attribute_name(key)
            attr_name = attr_name.removeprefix("threshold_")
            setattr(self, attr_name, attribute)
            self.subsystem.attribute_mapping[key] = attribute
