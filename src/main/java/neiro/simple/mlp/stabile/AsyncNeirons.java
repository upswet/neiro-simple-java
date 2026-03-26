package neiro.simple.mlp.stabile;

import neiro.simple.mlp.stabile.model.Neiron;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**Выполнение асинхронных действий над нейронами*/
public class AsyncNeirons {

    private static ExecutorService executor;
    private static boolean batchFlg;

    /**Инициализация асинхронных обработчиков нейронов
     * @param pocessorCount - кол-во процессоров. Минус один означает взять текущее кол-во процессоров системы
     * @param batchFlag - если истина то разбиваем массив нейронов на пачки и асинхронно выполняем каждую из них, иначе каждый нейрон выполняется в отдельной асинхронной задаче*/
    public static void init(int pocessorCount, boolean batchFlag){
        executor = Executors.newFixedThreadPool(
                pocessorCount == -1 ? Runtime.getRuntime().availableProcessors() : pocessorCount,
                r -> {
                    Thread t = new Thread(r);
                    t.setDaemon(true);
                    return t;
                }
        );

        batchFlg = batchFlag;
    }

    /**Вычислить размер пачки в зависимости от числа процессоров и размера списка нейронов*/
    public static int calcBatchSize(int nCount){
        int cpus = Runtime.getRuntime().availableProcessors();
        int desiredTasks = cpus * 2;                     // множитель можно настроить
        int batchSize = (nCount + desiredTasks - 1) / desiredTasks;
        return Math.max(1, Math.min(batchSize, 1000));// дополнительно ограничим batchSize снизу и сверху
    }

    public static void asyncForEach(List<Neiron> neirons, Consumer<Neiron> action){
        if (batchFlg)
            asyncForEachBatch(neirons, action, calcBatchSize(neirons.size()));
        else
            asyncForEachNoBatch(neirons, action);
    }

    public static void asyncIndexedFor(int size, Consumer<Integer> action){
        if (batchFlg)
            asyncIndexedForBatch(size, action, calcBatchSize(size));
        else
            asyncIndexedForNoBatch(size, action);

    }

    /**
     * Асинхронно выполняет действие для каждого элемента списка и ждёт завершения всех задач.
     * @param neirons список элементов
     * @param action действие, применяемое к каждому Neiron
     */
    public static void asyncForEachNoBatch(List<Neiron> neirons, Consumer<Neiron> action) {
        if (neirons.isEmpty()) {
            return;
        }
        CountDownLatch latch = new CountDownLatch(neirons.size());
        for (Neiron neiron : neirons) {
            executor.submit(() -> {
                try {
                    action.accept(neiron);
                } catch (Exception e) {
                    // Логируем ошибку, чтобы не потерять, но не прерываем остальные задачи
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Waiting for async tasks was interrupted", e);
        }
    }

    /**
     * Асинхронно выполняет действие для каждого элемента списка с доступом к индексу и ждёт завершения всех задач.
     * @param size количество элементов
     * @param action действие, принимающее индекс и сам Neiron
     */
    public static void asyncIndexedForNoBatch(int size, Consumer<Integer> action) {
        if (size==0) {
            return;
        }
        CountDownLatch latch = new CountDownLatch(size);
        for (int i = 0; i < size; i++) {
            int index = i;
            executor.submit(() -> {
                try {
                    action.accept(index);
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Waiting for async tasks was interrupted", e);
        }
    }

    /**
     * Асинхронно выполняет действие для каждого элемента списка, группируя элементы в пакеты.
     * @param neirons   список элементов
     * @param action    действие, применяемое к каждому Neiron
     * @param batchSize размер пакета (количество нейронов, обрабатываемых в одной задаче)
     */
    public static void asyncForEachBatch(List<Neiron> neirons, Consumer<Neiron> action, int batchSize) {
        if (neirons.isEmpty()) return;
        if (batchSize <= 0) batchSize = 1;

        int totalTasks = (neirons.size() + batchSize - 1) / batchSize;
        CountDownLatch latch = new CountDownLatch(totalTasks);

        for (int start = 0; start < neirons.size(); start += batchSize) {
            int end = Math.min(start + batchSize, neirons.size());
            List<Neiron> batch = neirons.subList(start, end);
            executor.submit(() -> {
                try {
                    for (Neiron neiron : batch) {
                        action.accept(neiron);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Waiting for async tasks was interrupted", e);
        }
    }

    /**
     * Асинхронно выполняет действие для каждого элемента списка с доступом к индексу, группируя элементы в пакеты.
     * @param size   количество элементов
     * @param action    действие, принимающее индекс и сам Neiron
     * @param batchSize размер пакета (количество нейронов, обрабатываемых в одной задаче)
     */
    public static void asyncIndexedForBatch(int size, Consumer<Integer> action, int batchSize) {
        if (size==0) return;
        if (batchSize <= 0) batchSize = 1;

        int totalTasks = (size + batchSize - 1) / batchSize;
        CountDownLatch latch = new CountDownLatch(totalTasks);

        for (int start = 0; start < size; start += batchSize) {
            int end = Math.min(start + batchSize, size);
            int finalStart = start;
            executor.submit(() -> {
                try {
                    for (int i = 0; i <end-finalStart; i++) {
                        int globalIndex = finalStart + i;
                        action.accept(globalIndex);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Waiting for async tasks was interrupted", e);
        }
    }

    public static void shutdown() {
        executor.shutdown();
    }



    //public static void shutdown() {executor.shutdown();}
}