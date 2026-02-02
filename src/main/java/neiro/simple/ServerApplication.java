package neiro.simple;

import lombok.SneakyThrows;
import neiro.simple.mlp.*;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static neiro.simple.mlp.Net.*;

@SpringBootApplication
public class ServerApplication {
	public static void main(String[] args) {
		//SpringApplication.run(ServerApplication.class, args);

		//ExampleMLP.xor();
		ExampleMLP.mnist();
	}
}
